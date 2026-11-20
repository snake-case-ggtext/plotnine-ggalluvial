# plotnine-ggalluvial

A clean-room [plotnine](https://plotnine.org/) port of R's `ggalluvial` package for drawing alluvial and Sankey-style flow diagrams.

## Implemented API

- **Geoms**: `geom_alluvium`, `geom_stratum`, `geom_flow`
- **Stats**: `stat_alluvium`, `stat_stratum`, `stat_flow` (registered automatically so `geom_text(aes(label=after_stat("stratum")), stat="stratum")` and `geom_label(..., stat="stratum")` work natively)
- **Data formats**: Wide (`axis1`, `axis2`, ..., `y`) and long/lodes (`x`, `stratum`, `alluvium`, `y`)
- **Curve types**: `xspline` (default, with `knot_pos` & `knot_prop`), `linear`, `cubic`, `quintic`, `sine`, `arctangent`/`arctan`, `sigmoid`
- **Ordering & binding**: `decreasing`, `reverse`, `lode_guidance` (`zigzag`, `zagzig`, `frontback`, `backfront`, `forward`, `backward`), `aes_bind` (`none`, `flows`, `alluvia`), `aes_flow` (`forward`, `backward`)
- **Layouts**: Multi-panel facets (`facet_wrap`, `facet_grid`) and `coord_flip()`

## Example

```python
from plotnine import aes, after_stat, geom_text, ggplot
from plotnine_ggalluvial import geom_alluvium, geom_stratum

p = (
    ggplot(df, aes(y="freq", axis1="axis1", axis2="axis2", axis3="axis3"))
    + geom_alluvium(aes(fill="cohort"))
    + geom_stratum()
    + geom_text(aes(label=after_stat("stratum")), stat="stratum")
)
```
