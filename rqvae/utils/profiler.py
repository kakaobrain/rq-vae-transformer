

class Profiler:
    opts_model_size = {'trainable-only', 'transformer-block-only'}

    def __init__(self, logger):
        self._logger = logger

    def get_model_size(self, model, opt=None):
        if opt is None:
            self._logger.info(
                "[OPTION: ALL] #parameters: %.4fM", sum(p.numel() for p in model.parameters()) / 1e6
            )
        else:
            assert opt in self.opts_model_size, f'{opt} is not in {self.opts_model_size}'

            if opt == 'trainable-only':
                self._logger.info(
                    "[OPTION: %s] #parameters: %.4fM", opt,
                    sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
                )
            else:
                if hasattr(model, 'blocks'):
                    self._logger.info(
                        "[OPTION: %s] #parameters: %.4fM", opt,
                        sum(p.numel() for p in model.blocks.parameters()) / 1e6
                    )
