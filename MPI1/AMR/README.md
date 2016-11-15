FINE_GRAIN: all work configurations are load balanced as well as possible, 
   regardless of communication involved, as follows: The background grid is
   split completely and evenly  among all participating ranks. When a 
   refinement comes into existence, it is split into a number of pieces equal 
   to the number of ranks, and each piece is assigned to a distinct rank 
   without regard for locality. 
NO_TALK: this strategy minimizes communication, as follows: The background 
   grid is split completely and evenly among all participating ranks. When a 
   refinement comes into existence, it is split into pieces that exactly 
   coincide with pieces of the background grid assigned to individual ranks. 
   Each refinement piece is assigned to the same rank that owns the underlying 
   piece of the background grid.
HIGH_WATER: the background grid and one refinement together are divided 
   statically as evenly as possible among the ranks. Each rank receives
   exactly one grid or one piece of a grid at a time. This means the code
   will not work for a single rank; this case is captured by the serial code.
   The decomposition and assignment of pieces of BG to the ranks is static.
   Refinements are partitioned identically and statically as well.
AMNESIA: each configuration of BG and refinements is partitioned as evenly as
   possible, such that each rank receives one (chunk of) a grid. Because 
   the refinements are all of the same size, this means that they are all
   partitioned identically. But there will be two different partitionings
   of the BG, one in the presence of a refinement, and one without. 
