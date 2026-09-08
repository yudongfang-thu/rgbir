    def _setup_train(self):
        """Configure model, optimizer, dataloaders, and training utilities before the training loop."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        # channels_last (NHWC) is CUDA-only: lossless and Tensor-Core friendly there, but numerically wrong
        # on MPS and no benefit on CPU
        if self.args.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        elif self.args.channels_last:
            LOGGER.warning(f"'channels_last=True' is only supported on CUDA, ignoring on '{self.device.type}'.")
        self.set_model_attributes()

        # Compile model (knowledge distillation runs the wrapped model eagerly and relies on
        # find_unused_parameters under DDP for the frozen teacher, so disable compilation when distilling)
        if self.args.distill_model is not None and self.args.compile:
            LOGGER.warning("'compile' is not supported with knowledge distillation and will be disabled.")
            self.args.compile = False
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        if isinstance(unwrap_model(self.model), DistillationModel):
            freeze_layer_names.append("teacher_model.")
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True
        if not any(v.requires_grad for v in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.args.freeze}' froze the entire model with no trainable parameters left. "
                f"Reduce 'freeze' or pass a list of specific layer indices."
            )

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            self.amp = self.amp.int()  # gloo errors with boolean
            dist.broadcast(self.amp, src=0)  # broadcast from rank 0 to all other ranks
        self.amp = bool(self.amp)  # as boolean
        if self.device.type == "npu":
            import torch_npu

            self.scaler = torch_npu.npu.amp.GradScaler(enabled=self.amp)
        else:
            self.scaler = (
                torch.amp.GradScaler(self.device.type if self.device.type == "xpu" else "cuda", enabled=self.amp)
                if TORCH_2_4
                else torch.cuda.amp.GradScaler(enabled=self.amp)
            )
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # resume training would directly load DistillationModel so check here
        if self.args.distill_model is not None and not isinstance(unwrap_model(self.model), DistillationModel):
            self.model = DistillationModel(student_model=self.model, teacher_model=self.args.distill_model)
        if self.world_size > 1:
            # static_graph=True permits params used >1 time per forward (e.g. flow_model in
            # o2m+o2o pose loss branches) under torch.compile.
            ddp_kwargs = {"static_graph": bool(self.args.compile)} if TORCH_1_11 else {}
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                find_unused_parameters=not bool(self.args.compile),
                **ddp_kwargs,
            )

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()
        self._build_train_pipeline()
        self.validator = self.get_validator()
        self.ema = ModelEMA(self.model)
        self.set_class_weights()  # compute class weights after dataloader is ready
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")
