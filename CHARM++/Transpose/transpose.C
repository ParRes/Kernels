#include "transpose.decl.h"
#include <par-res-kern_general.h>

#define A(i,j)        A_p[(i+istart)+order*(j)]
#define B(i,j)        B_p[(i+istart)+order*(j)]

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int order; // array size
/*readonly*/ int num_chares;
/*readonly*/ int overdecomposition; 
/*readonly*/ int iterations;
/*readonly*/ int Block_order;
/*readonly*/ int Tile_order;
/*readonly*/ int tiling;
/*readonly*/ int Colblock_size;
/*readonly*/ int Block_size;
/*readonly*/ long bytes;
class blockMsg : public CMessage_blockMsg {
public:
  int blockID;
  double *blockData;
  blockMsg(int _t) : blockID(_t) {
  }
};

class Main : public CBase_Main
{

public:
    CProxy_Transpose array;

    Main(CkArgMsg* cmdlinearg) {
        CkPrintf("Parallel Research Kernels Version %s\n", PRKVERSION);
        CkPrintf("Charm++ transpose execution\n");

        if (cmdlinearg->argc != 5) {
          CkPrintf("%s <#iterations> <matrix order> <tile size><overdecomposition factor>\n",
          cmdlinearg->argv[0]); CkExit();
        }

        // store the main proxy
        mainProxy = thisProxy;

        iterations = atoi(cmdlinearg->argv[1]);
        if (iterations < 1) {
          CkPrintf("ERROR: #iterations must be positive: %d\n", iterations);
          CkExit();
        }
        order = atoi(cmdlinearg->argv[2]);
        if (order < CkNumPes()) {
          CkPrintf("ERROR: Matrix order %d smaller than #PEs %d\n", order, CkNumPes());
          CkExit();
        }

        Tile_order = atoi(cmdlinearg->argv[3]);
        if (Tile_order < 1) {
          CkPrintf("ERROR: Tile size must be positive: %d \n", Tile_order);
          CkExit();
        }

        overdecomposition = atoi(cmdlinearg->argv[4]);
        if (overdecomposition<1) {
          CkPrintf("ERROR: Overdecomposition factor must be positive: %d\n", overdecomposition);
          CkExit();
        }

        num_chares = CkNumPes()*overdecomposition;
        if (!(order/num_chares)) {
          CkPrintf("ERROR: Matrix order %d smaller than #chares %d\n", order, num_chares);
          CkExit();
        }
        if (order%num_chares) {
          CkPrintf("ERROR: Matrix order %d not multiple of #chares %d\n", order, num_chares);
          CkExit();
        }

        Block_order = order/num_chares;
        Colblock_size = order * Block_order;
        Block_size  = Block_order * Block_order;

        tiling = (Tile_order > 0) && (Tile_order < order);
        bytes = 2 * sizeof(double) * order * order;

        // print info
        CkPrintf("Number of Charm++ PEs = %d\n", CkNumPes());
        CkPrintf("Overdecomposition     = %d\n", overdecomposition);
        CkPrintf("Matrix order          = %d\n", order);
        CkPrintf("Tile size             = %d\n", Tile_order);
        CkPrintf("Number of iterations  = %d\n", iterations);

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
  int iter, phase, colstart;
  double result, local_error;
  int send_to, recv_from;
  double * RESTRICT A_p, * RESTRICT B_p, * RESTRICT Work_in_p, * RESTRICT Work_out_p;
  double startTime, endTime;
  // Constructor, initialize values
  Transpose() {
    // allocate two dimensional array
    A_p        = new double[Colblock_size];
    B_p        = new double[Colblock_size];
    if (!A_p || !B_p ) { 
      CkPrintf("Could not allocate memory for matrix blocks\n");
      CkExit();
    }

    /* set value of starting column for this chare                      */
    colstart = thisIndex*Block_order;

    /* Fill the original column matrix in A.                            */
    int istart = 0;  
    for (int j=0;j<Block_order;j++) for (int i=0;i<order; i++) {
      A(i,j) = (double) (order*(j+colstart) + i);
    }

    /*  Set the transpose matrix to zero                                */
    for (int i=0;i<Colblock_size; i++) B_p[i] = 0.0;
    
  }

  // a necessary function, which we ignore now. If we were to use load balancing 
  // and migration this function might become useful
  Transpose(CkMigrateMessage* m) {}

  ~Transpose() { 
    delete [] A_p;
    delete [] B_p;
  }

  // Perform one matrix block worth of work
  void diagonal_transpose() {
    int istart = colstart; 
    if (!tiling) {
      for (int i=0; i<Block_order; i++) 
        for (int j=0; j<Block_order; j++) {
          B(j,i) += A(i,j);
          A(i,j) += 1.0;
	}
    }
    else {
      for (int i=0; i<Block_order; i+=Tile_order) 
        for (int j=0; j<Block_order; j+=Tile_order) 
          for (int it=i; it<MIN(Block_order,i+Tile_order); it++)
            for (int jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
              B(jt,it) += A(it,jt); 
              A(it,jt) += 1.0;
            }
    }
  }

  void nondiagonal_transpose(int send_to) {
      blockMsg *msg = new (Block_size) blockMsg(thisIndex);
      if (!msg) {
        CkPrintf("Could not allocate space for message\n");
        CkExit();
      }

      int istart = send_to*Block_order; 
      if (!tiling) {
        for (int i=0; i<Block_order; i++) 
          for (int j=0; j<Block_order; j++){
	    msg->blockData[j+Block_order*i] = A(i,j);
            A(i,j) += 1.0;
	  }
      }
      else {
        for (int i=0; i<Block_order; i+=Tile_order) 
          for (int j=0; j<Block_order; j+=Tile_order) 
            for (int it=i; it<MIN(Block_order,i+Tile_order); it++)
              for (int jt=j; jt<MIN(Block_order,j+Tile_order);jt++) {
                msg->blockData[it+Block_order*jt] = A(jt,it); 
                A(jt,it) += 1.0;
	      }
      }
      CkSetRefNum(msg,iter*num_chares);
      thisProxy(send_to).receiveBlock(msg);
  }

  // The final step is to receive the transposed block and store it in the proper place    
  void processBlock(blockMsg *msg) {

    int istart = msg->blockID*Block_order; 
    /* scatter received block to transposed matrix; no need to tile */
    for (int j=0; j<Block_order; j++)
      for (int i=0; i<Block_order; i++) 
        B(i,j) += msg->blockData[i+Block_order*j];
  
    delete msg;
  }

  void compute_local_error() {

    local_error = 0.0;
    int istart = 0;
    double addit = ((double)(iterations+1) * (double) (iterations))/2.0;
    for (int j=0;j<Block_order;j++) for (int i=0;i<order; i++) {
      local_error += ABS(B(i,j) - (double)((order*i + j+colstart)*(iterations+1) +addit));
    }
  }

};
#include "transpose.def.h"
