"""Restore the CFT block name in the author's published legacy checkpoint.

The released pickle uses TransformerBlock for the CFT attention block, whereas
the public repository uses myTransformerBlock for that structure. This is an
explicit runtime adaptation, not a claim that the checkpoint's training source
has been recovered. It changes no tensor or module topology.
"""
def restore_cft_block_names(model):
    from models.common import TransformerBlock, myTransformerBlock
    expected = {'ln_input', 'ln_output', 'sa', 'mlp'}
    parameter_ids = {n: id(p) for n, p in model.named_parameters()}
    buffer_ids = {n: id(b) for n, b in model.named_buffers()}
    restored = []
    untouched = []
    for name, module in model.named_modules():
        if type(module) is not TransformerBlock:
            continue
        if set(module._modules) != expected:
            untouched.append(name)
            continue
        assert not module._parameters and not module._buffers
        assert '.trans_blocks.' in name, name
        assert not any(hasattr(module, x) for x in ('conv', 'linear', 'tr', 'c2'))
        module.__class__ = myTransformerBlock
        restored.append(name)
    assert restored, 'No legacy CFT blocks found; do not silently apply a different repair'
    assert parameter_ids == {n: id(p) for n, p in model.named_parameters()}
    assert buffer_ids == {n: id(b) for n, b in model.named_buffers()}
    return dict(
        adaptation='conditional legacy TransformerBlock to author myTransformerBlock class rebinding',
        restored_blocks=restored, untouched_other_transformer_blocks=untouched,
        parameter_and_buffer_objects_unchanged=True,
        random_parameters_created=False,
        original_author_source_modified=False,
        historical_source_limit='Both public common.py commits already use myTransformerBlock; exact checkpoint training source unavailable',
        equivalence_scope='Stored child modules match current author CFT class; no claim of numerical equivalence to unavailable training code')
