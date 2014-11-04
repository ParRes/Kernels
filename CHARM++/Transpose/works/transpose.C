#include "transpose.decl.h"
#define MIN(i,j)      ((i)<(j) ? (i) : (j))
// Constant to shift column index 
#define  COL_SHIFT  1.00
// Constant to shift row index 
#define  ROW_SHIFT  0.1  
#define A(i,j)              A[(i)+Block_order*(j)]
#define B(i,j)              B[(i)+Block_order*(j)]
#define Work(i,j)           Work_p[(i)+Block_order*(j)]
#define Orig_Colblock(i,j)  Orig_Colblock_p[(i)*Block_order+j]
#define Trans_Colblock(i,j) Trans_Colblock_p[(i)*Block_order+j]


// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int order; // array size
/*readonly*/ int num_chares;
/*readonly*/ int overdecomposition; 
/*readonly*/ int maxiterations;
/*readonly*/ int Block_order;
/*readonly*/ int Tile_order;
/*readonly*/ int Colblock_size;
/*readonly*/ int Block_size;
/*readonly*/ double startTime;
/*readonly*/ double endTime;
/*readonly*/ long bytes;

class tileMsg : public CMessage_tileMsg {
public:
  int tile_ID;
  double *tiledata;
  tileMsg(int _t) : tile_ID(_t) {
  }
};

class Main : public CBase_Main
{

public:
    CProxy_Transpose array;

    Main(CkArgMsg* cmdlinearg) {
        if (cmdlinearg->argc != 5) {
          CkPrintf("%s <#iterations> <matrix order> <tile order><overdecomposition factor>\n",
          cmdlinearg->argv[0]); CkAbort("Abort");
        }

        // store the main proxy
        mainProxy = thisProxy;

        maxiterations = atoi(cmdlinearg->argv[1]);
        if (maxiterations < 1) {
          CkAbort("ERROR: #iterations must be positive");
        }
        order = atoi(cmdlinearg->argv[2]);
        if (order < CkNumPes()) 
          CkAbort("ERROR: Horizontal grid size smaller than #PEs\n");

        Tile_order = atoi(cmdlinearg->argv[3]);
        if (Tile_order < 1) 
          CkAbort("ERROR: Matrix order size must be positive\n");

        overdecomposition = atoi(cmdlinearg->argv[4]);
        if (overdecomposition<1)
          CkAbort("ERROR: Overdecomposition factor must be positive\n");

        num_chares = CkNumPes()*overdecomposition;
        if (!(order/num_chares))
          CkAbort("ERROR: Matrix order smaller than #PEs*overdecomposition factor\n");
        if (order%num_chares)
          CkAbort("ERROR: Matrix order must be multiple of #PEs*overdecomposition factor\n");
        Block_order = order/num_chares;
        Colblock_size = order * Block_order;
        Block_size  = Block_order * Block_order;

        bytes = 2 * sizeof(double) * order * order;

        // print info
        CkPrintf("Charm++ transpose execution\n");
        CkPrintf("Number of processes  = %d\n", CkNumPes());
        CkPrintf("Overdecomposition    = %d\n", overdecomposition);
        CkPrintf("Matrix order         = %d\n", order);
        CkPrintf("Tile order           = %d\n", Tile_order);
        CkPrintf("Number of iterations = %d\n", maxiterations);

        // Create new array of worker chares
        array = CProxy_Transpose::ckNew(num_chares);

        //Start the computation
	array.run();
    }

    // One worker reports back to here when it completes the workload
    void report(double result) {

      double epsilon = 1.e-8, avgtime;
      if (result < epsilon) {
        CkPrintf("Solution validates\n");
        avgtime = (endTime-startTime)/(double)maxiterations;
        CkPrintf("Rate (MB/s): %lf, Avg time (s): %lf\n",1.0E-06*bytes/avgtime, avgtime);
      }
      else                  
	CkPrintf("Solutions does not validate; diff = %e, threshold = %e\n",
                 result, epsilon);
      CkExit();
    }

};

class Transpose: public CBase_Transpose {
  Transpose_SDAG_CODE

public:
  int iterations, phase;
  double result, local_error;
  int send_to, recv_from;
  double *Orig_Colblock_p, *Trans_Colblock_p, *Work_p;
  double **Orig_Block_p, **Trans_Block_p;

  // Constructor, initialize values
  Transpose() {
    int i, j;
      
    // allocate two dimensional array
    Orig_Colblock_p  = new double[Colblock_size];
    Trans_Colblock_p = new double[Colblock_size];
    Work_p           = new double[Block_size];
    if (! Orig_Colblock_p || !Trans_Colblock_p || !Work_p) {
      CkAbort("Could not allocate memory for matrix blocks\n");
  }

    /* Fill the original column matrix in Orig_Colblock.                */
    for (i=0;i<order; i++) for (j=0;j<Block_order;j++) {
	Orig_Colblock(i,j) = COL_SHIFT*(thisIndex*Block_order+(j+1)) + ROW_SHIFT*(i+1);
    }

    /*  Set the transpose matrix to a known garbage value.              */
    for (i=0;i<Colblock_size; i++) Trans_Colblock_p[i] = -1.0;

    /*********************************************************************
     ** create entry points to the Blocks within Colblocks
     ********************************************************************/

    Orig_Block_p =  new double*[num_chares];
    if (!Orig_Block_p) CkAbort("Error allocating space for block pointers\n");
    Trans_Block_p =  new double*[num_chares];
    if (!Trans_Block_p) CkAbort("Error allocating space for block pointers\n");
    for (i=0; i<num_chares; i++) {
      Orig_Block_p[i]  = Orig_Colblock_p  + i*Block_size;
      Trans_Block_p[i] = Trans_Colblock_p + i*Block_size;
    }
  }

  // a necessary function, which we ignore now. If we were to use load balancing 
  // and migration this function might become useful
  Transpose(CkMigrateMessage* m) {}

  ~Transpose() { 
    delete [] Orig_Block_p;
    delete [] Trans_Block_p;
    delete [] Orig_Colblock_p;
    delete [] Trans_Colblock_p;
    delete [] Work_p;
  }

  // Perform one matrix tile worth of work
  // The first step is to do the local transpose of the tile
  void local_transpose(int blockIndex) {
    int i, j, it, jt;
    double *A, *B;
    A = Orig_Block_p[blockIndex];
    if (blockIndex==thisIndex) B = Trans_Block_p[thisIndex];
    else                       B = Work_p;
    // tile only if the tile size is smaller than the matrix block  
    if (Tile_order < Block_order) {
      for (i=0; i<Block_order; i+=Tile_order) 
        for (j=0; j<Block_order; j+=Tile_order) 
          for (it=i; it<MIN(Block_order,i+Tile_order); it++)
            for (jt=j; jt<MIN(Block_order,j+Tile_order);jt++)
              B(it,jt) = A(jt,it); 
    }
    else {
      for (i=0;i<Block_order; i++) 
        for (j=0;j<Block_order;j++) {
          B(i,j) = A(j,i);
	}
         
    }
  }

  // The next step is to send the tile to the right destination
  void sendTile(int destination) {

    tileMsg *msg = new (Block_size) tileMsg(thisIndex);
    CkSetRefNum(msg, phase+iterations*num_chares);
    memcpy(msg->tiledata, Work_p, Block_size*sizeof(double));
    thisProxy(destination).receiveTile(msg);
  }

  // The final step is to receive the transposed tile and store it in the right place    
  void processTile(tileMsg *msg) {

    memcpy(Trans_Block_p[msg->tile_ID],msg->tiledata,Block_size*sizeof(double));
    delete msg;
  }

  void compute_local_error() {

    double diff;
    local_error = 0.0;
    for (int i=0;i<order; i++) {
      for (int j=0;j<Block_order;j++) {
        diff = Trans_Colblock(i,j)-(COL_SHIFT*((i+1)) + ROW_SHIFT*(thisIndex*Block_order+j+1));
        local_error += diff*diff;
      }
    }
  }

};

#include "transpose.def.h"
