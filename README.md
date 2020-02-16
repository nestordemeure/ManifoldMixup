# Manifold Mixup V1

Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) for [fastai V1](https://docs.fast.ai/index.html) based on [Shivam Saboo](https://github.com/shivamsaboo17)'s [pytorch implementation](https://github.com/shivamsaboo17/ManifoldMixup) of manifold mixup, fastai's input mixup [implementation](https://docs.fast.ai/callbacks.mixup.html) plus some personnal improvements/variants.

This package provides two additional methods to the fastai learner :
- `.manifold_mixup()` which implements [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf)
- `.output_mixup()` which implements a variant that does the mixup only on the output of the last layer (this was shown to be more performant on a [benchmark](https://forums.fast.ai/t/mixup-data-augmentation/22764/72) and an independant [blogpost](https://medium.com/analytics-vidhya/better-result-with-mixup-at-final-layer-e9ba3a4a0c41))

**Note:** For a [fastai V2](http://dev.fast.ai/) compatible version, see the [V2 repository](https://github.com/nestordemeure/ManifoldMixupV2).

## Usage

To use manifold mixup, you just need to call a method, either `manifold_mixup` or `output_mixup`, on your learner (for a minimal demonstration, see the [Demo notebook](https://github.com/nestordemeure/ManifoldMixup/blob/master/Demo.ipynb)):

```python
learner = Learner(data, model).manifold_mixup()
learner.fit(8)
```

The `manifold_mixup` method takes four parameters :
- `alpha=0.4` parameter of the beta law used for sampling the interpolation weight
- `use_input_mixup=True` do you want to apply mixup to the inputs
- `module_list=None` can be used to pass an explicit list of target modules
- `stack_y=True` do you want to perform the combinaison after the evaluation of the loss function (good for classification) or directly on the raw targets (good for regression)

The `output_mixup` variant takes only the `alpha` parameter.

## Notes

### Which modules will be intrumented ?

`manifold_mixup` tries to establish a sensible list of modules on which to apply mixup:
- it uses a user provided `module_list` if possible
- otherwise it uses only the modules wrapped with `ManifoldMixupModule`
- if none are found, it defaults to modules with `Block` in their name (targetting mostly resblocks)
- finaly, if needed, it defaults to all modules that are not included in the `non_mixable_module_types` list

The `non_mixable_module_types` list contains mostly recurrent layers but you can add elements to it in order to define module classes that should not be used for mixup (*do not hesitate to create an issue or start a PR to add common modules to the default list*).

### When can I use OutputMixup ?

`output_mixup` applies the mixup directly to the output of the last layer.
This only works if the loss function contains something like a softmax (and not when it is directly used as it is for regression).

### A note on skip-connections / residual-blocks

`manifold_mixup` (this does not apply to `output_mixup`) is greatly degraded when applied *inside* a residual block.
This is due to the mixed-up values becoming incoherent with the output of the skip connection (which have not been mixed).

While this implementation is equiped to work around the problem for U-Net and ResNet like architectures, you might run into problems (negligeable improvements over the baseline) with other network structures.
In which case, the best way to apply manifold mixup would be to manually select the modules to be instrumented.

*For more unofficial fastai extensions, see the [Fastai Extensions Repository](https://github.com/nestordemeure/fastai-extensions-repository).*
