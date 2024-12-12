# ROMAN Demo

Using `demo.py`, the full ROMAN pipeline can be run on a multi-robot dataset, or mapping, loop closures, or pose graph optimization can be separately run.

After following the [install instructions](../README.md/#install) and the [demo setup instructions](../README.md/#demo), the demo can be run with the following command:

```
cd roman
mkdir demo_output
python3 demo/demo.py \
    -r sparkal1 sparkal2 -e ROBOT \
    -p demo/params/demo \
    -o demo_output
```

## Custom Data