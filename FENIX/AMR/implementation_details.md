Refinements are partitioned in terms of true indices (i.e. refinement
indices).  The refinement partitions are initialized based on BG
values, so for each refinement we need the corresponding BG
values. Here it matters whether the BG is truly coarser than the
refinement (TRUE_REF). If so, at least some of the refinement
partition boundaries do not coincide with BG points, because the
refinement patch partitions form a disjoint covering. To interpolate
the refinement boundary values requires some BG values beyond the
refinement partition, even for the NO_TALK balancer. 
For FINE_GRAIN and HIGH_WATER we take the following approach.
Determine the BG patch boundaries required for initialization of the
refinement partition.  These boundaries will overlap if TRUE_REF. All
ranks request (MPI_Alltoallv) that data from the ranks that own it
directly, so that the overlap data is immediately available.
For NO_TALK we take a different approach. At the beginning of each time 
step we first do the exchange of BG values among BG partitions. This 
leads to an overlap of RADIUS grid cells between BG partitions. we can
take these enlarged BG partitions into account when ranks request BG 
values from other ranks. Consequently, NO_TALK truly does not need to 
communicate to initialize the refinements (the BG exchange does not 
constitute a refinement initialization overhead, because it needs to
happen anyway).
ADDENDUM: For consistency, we will ALWAYS do the exchange of BG values 
among BG partitions at the beginning of each time step, not just for
the NO_TALK scenario, but we will not make use of this fact for
fetching BG data in the other two scenarios. The reason is that it
introduces an ambiguity regarding which rank should furnish BG data
in overlap regions.

Nomenclature of variables defining grid and patch boundaries (all
variables have an i- and j-variant, as well as a start- and end variant)
Prefix G_: indicates that this boundary holds globally, i.e. not tied
           to the calling rank
Prefix L_: indicates that this boundary holds locally, i.e. tied to
           the calling rank
Suffix _gross: indicates a refinement patch boundary that has been
           padded to include the underlying BG points needed for
           interpolation
Suffix _bg: indicates a boundary of a BG patch
Suffix _r:  indicates a boundary of a refinement patch
Suffix _true: indicates a refinement patch boundary in terms of refinement
           indices, i.e. taking into account expansion due to refinement,
           and shifted so that (0,0) corresponds to the left bottom corner 
           of the refinement
L_istart_bg     = staring index of BG partition owned by calling rank
G_istart_r[g]   = global starting index of refinement g, based on BG 
                 indices
L_istart_r[g]   = starting index of patch of refinement g, based on BG 
                 indices, owned by calling rank
L_istart_r_gross[g] = starting index of patch of refinement g, based on BG 
                 indices, owned by calling rank, padded to include BG points
                 necessary for interpolation
L_istart_r_true[g] = starting index of patch of refinement g, relative to
                 refinement, owned by calling rank (includes refinement expansion)
L_istart_r_true_gross[g]   = starting index of patch of refinement g, relative
                 to refinement, owned by calling rank, padded to include BG points
                 necessary for interpolation (includes refinement expansion)
                   
Steps:
1: Compute refinement patch boundaries based on true, refined indices. 
   This is ambiguous for NO_TALK if TRUE_REF, because a small shift in 
   patch boundaries would still not require communications. We use the
   following disambiguation. Partition BG points of the refinement patch 
   in the same way as the BG itself. Place true refinement boundaries
   in the middle between BG points. This creates the best load balance,
   and also avoids as much as possible very small refinement patch slivers
   that could create problems with ghost point acquisition.
   one point to the left and/or bottom, unless the partition already 
   borders on the respective edge of the partition; this creates a 
   one-point BG overlap between adjacent refinement partitions. All 
   true refinement points that lie within the thus created refinement 
   partition, except any points that coincide with the augmented BG
   boundary, are assigned to the rank that owns the augmented partition. 
   NO_TALK
   BG:          |       X       X       X       X       X       |
   assignment   a       a       b       b       b       c       c
   refinement   |.......x.......x.......x.......x.......x.......| 
   assignment   aaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbccccccccccccc
   allocation   aaaaaaaaaaaaaaaaa
                        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
                                                ccccccccccccccccc

   Legend: |   = refinement patch boundary point
           X   = refinement patch BG interior point
           .   = true refinement point that is not a BG point
           x   = true refinement point that is also a BG point
           a,b = point assigned to rank a,b
   In the case of HIGH_WATER and FINE_GRAIN we follow a similar process
   for memory allocation for the refinements, but only in terms of
   accommodating enough space for overlaps of BG points. The actual 
   assignment of true refinement points to ranks does not take BG cells
   into account. 

   FINE_GRAIN (assuming a total of 4 ranks in the horizontal direction)
              Sets {a,b,c} and {p,q,r,s} generally overlap.
   BG:          |       X       X       X       X       X       |
   assignment   a       a       b       b       b       c       c
   refinement   |.......x.......x.......x.......x.......x.......| 
   assignment   pppppppppppppqqqqqqqqqqqqrrrrrrrrrrrrssssssssssss
   allocation   ppppppppppppppppp
                        qqqqqqqqqqqqqqqqq
                                        rrrrrrrrrrrrrrrrr
                                                sssssssssssssssss
   
   HIGH_WATER (assuming a total of 4 ranks in the horizontal direction
              Sets {a,b,c} and {p,q,r,s} do not overlap.
   BG:          |       X       X       X       X       X       |
                a       a       b       b       b       c       c
   refinement   |.......x.......x.......x.......x.......x.......| 
   assignment   pppppppppppppqqqqqqqqqqqqrrrrrrrrrrrrssssssssssss
   allocation   ppppppppppppppppp
                        qqqqqqqqqqqqqqqqq
                                        rrrrrrrrrrrrrrrrr
                                                sssssssssssssssss
   Rationale for differences in assignment strategies: NO_TALK is
   based on rank assignment of BG points, but FINE_GRAIN and 
   HIGH_WATER are not. It is possible in case of FINE_GRAIN that there
   are not enough BG point in each refinement to give all ranks some
   work, even if they have enough. For HIGH_WATER we want to have the 
   most even distribution of work among the ranks.
   
2. Compute BG patch boundaries needed to initialize refinement patches.
   These will overlap if TRUE_REF. => i/jstart/end_bgl[g]
3. Allocate refinement patches such that they cover complete BG cells
   computed in step 2. 
4. For NO_TALK, copy BG values directly into corresponding locations
   of refinement patches, without communications. For HIGH_WATER and
   FINE_GRAIN, we need to fetch the BG values (multiple communications)
   and then store into corresponding locations of refinement patches.