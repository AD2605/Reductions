A simple Device Callable, highly performant Sum reduction implemented in cuda at various levels, Grid, Cluster, Block, Warp and Thread.

Achieves ~95-97% of theoritical bandwidth across architectures. 
For now, only works for multiples of 4, though boundary checking can be easily
added
