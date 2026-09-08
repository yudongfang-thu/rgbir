def check_amp(model):
    """Check the PyTorch Automatic Mixed Precision (AMP) functionality of a YOLO model.

    If the checks fail, it means there are anomalies with AMP on the system that may cause NaN losses or zero-mAP
    results, so AMP will be disabled during training.

    Args:
        model (torch.nn.Module): A YOLO model instance.

    Returns:
        (bool): Returns True if the AMP functionality works correctly with YOLO model, else False.

    Examples:
        >>> from ultralytics import YOLO
        >>> from ultralytics.utils.checks import check_amp
        >>> model = YOLO("yolo26n.pt").model.cuda()
        >>> check_amp(model)
    """
    from ultralytics.utils.torch_utils import autocast

    device = next(model.parameters()).device  # get model device
    prefix = colorstr("AMP: ")
    if device.type in {"cpu", "mps"}:
        return False  # AMP only used on accelerator devices
    elif device.type == "cuda":
        # GPUs that have issues with AMP
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)", re.IGNORECASE
        )

        gpu = torch.cuda.get_device_name(device)
        if bool(pattern.search(gpu)):
            LOGGER.warning(
                f"{prefix}checks failed ❌. AMP training on {gpu} GPU may cause "
                f"NaN losses or zero-mAP results, so AMP will be disabled during training."
            )
            return False

    def amp_allclose(m, im):
        """All close FP32 vs AMP results."""
        batch = [im] * 8
        imgsz = max(256, int(model.stride.max() * 4))  # max stride P5-32 and P6-64
        a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # FP32 inference
        with autocast(enabled=True, device=device.type):
            b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data  # AMP inference
        del m
        return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)  # close to 0.5 absolute tolerance

    im = ASSETS / "bus.jpg"  # image to check
    LOGGER.info(f"{prefix}running Automatic Mixed Precision (AMP) checks...")
    warning_msg = "Setting 'amp=True'. If you experience zero-mAP or NaN losses you can disable AMP with amp=False."
    try:
        from ultralytics import YOLO

        assert amp_allclose(YOLO("yolo26n.pt"), im)
        LOGGER.info(f"{prefix}checks passed ✅")
    except ConnectionError:
        LOGGER.warning(f"{prefix}checks skipped. Offline and unable to download YOLO26n for AMP checks. {warning_msg}")
    except (AttributeError, ModuleNotFoundError):
        LOGGER.warning(
            f"{prefix}checks skipped. "
            f"Unable to load YOLO26n for AMP checks due to possible Ultralytics package modifications. {warning_msg}"
        )
    except AssertionError:
        LOGGER.error(
            f"{prefix}checks failed. Anomalies were detected with AMP on your system that may lead to "
            f"NaN losses or zero-mAP results, so AMP will be disabled during training."
        )
        return False
    return True
