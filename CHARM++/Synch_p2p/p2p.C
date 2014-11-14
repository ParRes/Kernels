#include "p2p.decl.h"
#define EPSILON       1.e-8
#define ARRAY(i,j) vector[i+1+(j)*(width+1)]
#define MAX(i,j)      ((i)>(j) ? (i) : (j))
#define MIN(i,j)      ((i)<(j) ? (i) : (j))
#define ABS(x)        ((x)<0.0 ? (-1.0*(x)) : (x))

// See README for documentation

/*readonly*/ CProxy_Main mainProxy;
/*readonly*/ int n; // array size
/*readonly*/ int m; // array size
/*readonly*/ int overdecomposition; 
/*readonly*/ int maxiterations;

// specify the number of worker chares in each dimension
/*readonly*/ int num_chares;
/*readonly*/ double startTime, endTime;

class ghostMsg : public CMessage_ghostMsg {
public:
  double *gp;
  ghostMsg(){
  }
};

class cornerMsg : public CMessage_cornerMsg {
public:
  double *gp;
  cornerMsg(){
  }
};

class Main : public CBase_Main
{

public:
    CProxy_P2p array;

    Main(CkArgMsg* cmdlinearg) {
        if (cmdlinearg->argc != 5) {
          CkPrintf("%s <#iterations> <grid_size x> <grid_size y><overdecomposition factor>\n",
          cmdlinearg->argv[0]); CkExit();
        }

        // store the main proxy
        mainProxy = thisProxy;

        maxiterations = atoi(cmdlinearg->argv[1]);
        if (maxiterations < 1) {
          CkPrintf("ERROR: #iterations must be positive: %d", maxiterations);
          CkExit();
        }
        m = atoi(cmdlinearg->argv[2]);
        if (m < CkNumPes()) {
          CkPrintf("ERROR: Horizontal grid size %d smaller than #PEs %d\n", m, CkNumPes());
          CkExit();
        }

        n = atoi(cmdlinearg->argv[3]);
        if (n < 1) {
          CkPrintf("ERROR: Vertical grid size must be positive: %d\n", n);
          CkExit();
        }

        overdecomposition = atoi(cmdlinearg->argv[4]);
        if (overdecomposition<1) {
          CkPrintf("ERROR: Overdecomposition factor must be positive: %d\n", overdecomposition);
          CkExit();
        }

        int min_size = m/(CkNumPes()*overdecomposition);
        if (!min_size) {
          CkPrintf("ERROR: Horizontal grid size %d smaller than #PEs*overdecomposition factor %d\n",
		   m, CkNumPes()*overdecomposition);
          CkExit();
        }

        num_chares = CkNumPes()*overdecomposition;

        // print info
        CkPrintf("Charm++ pipeline execution on 2D grid\n");
        CkPrintf("Number of processes  = %d\n", CkNumPes());
        CkPrintf("Overdecomposition    = %d\n", overdecomposition);
        CkPrintf("Grid sizes           = %d,%d\n", m, n);
        CkPrintf("Number of iterations = %d\n", maxiterations);

        // Create new array of worker chares
        array = CProxy_P2p::ckNew(num_chares);
        // need to figure out how to trap errors in creating chare arrays
	//        if (!array) {
	//          CkPrintf("ERROR: Could not allocate space for chare array\n");
	//          CkExit();
	//        }

        //Start the computation
	array.run();
    }

    // One worker reports back to here when it completes the workload
    void report(double result) {
      double totalTime, flops, diff;
      double corner_val = (double) ((maxiterations+1)*(m+n-2));
      totalTime = endTime - startTime;
      flops = (double) (2*(n-1)) * (double) (m-1)*maxiterations;
      diff = ABS(result-corner_val);
      if (diff < EPSILON) {
        CkPrintf("Solution validates; ");
        CkPrintf("MFlops: %lf Avg time (s) %lf\n", flops/totalTime/1.e6, totalTime/maxiterations);
      }
      else {
        CkPrintf("Solution does not validate\n");
      }
      CkPrintf("Reference corner value: %lf, corner value: %lf, |diff|: %e \n", 
               corner_val, result, diff);
      CkExit();
    }

};

class P2p: public CBase_P2p {
  P2p_SDAG_CODE

public:
  int iterations;
  double result;
  int    offset, istart, iend, j; // global grid indices of strip
  int    width; 
  double *vector;

  // Constructor, initialize values
  P2p() {
    int i, iloc, leftover;
      
    /* compute amount of space required for input and solution arrays             */
    width = m/overdecomposition;
    leftover = m%overdecomposition;
    if (thisIndex < leftover) {
      istart = (width+1) * thisIndex; 
      iend = istart + width;
    }
    else {
      istart = (width+1) * leftover + width * (thisIndex-leftover);
      iend = istart + width - 1;
    }
    width = iend - istart + 1;

    // allocate two dimensional array
    vector = new double[n*(width+1)];
    if (!vector) {
      CkPrintf("ERROR: Char %d could not allocate array of size %d\n", thisIndex, n*(width+1));
      CkExit();
    }

    // initialize
    if (thisIndex == 0) for (j=0; j<n; j++) ARRAY(0,j) = (double) j;
    for(i=istart-1;i<=iend;i++) ARRAY(i-istart,0) = (double) i;
    if (thisIndex == 0) offset=1; else offset=0;
  }

  // a necessary function, which we ignore now. If we were to use load balancing 
  // and migration this function might become useful
    P2p(CkMigrateMessage* m) {}

    ~P2p() { 
      delete [] vector;
    }

    // Perform one grid line worth of work
    // The first step is to receive data from a left neighbor, if any
    void processGhost(ghostMsg *msg) {

      ARRAY(-1,j) = msg->gp[0];
      delete msg;
    }

    // do the actual work
    void compute() {
      int iloc;
      for (int i=istart+offset,iloc=offset; i<=iend; i++,iloc++) {
        ARRAY(iloc,j) = ARRAY(iloc-1,j) + ARRAY(iloc,j-1) - ARRAY(iloc-1,j-1);
      }
    }

    // The final step is to send the local state to the neighbors
    void pass_baton(void) {

      // Send my right edge
      if (thisIndex < num_chares-1) {
          ghostMsg *msg = new (1) ghostMsg();
          CkSetRefNum(msg, j+iterations*(n-1));
          msg->gp[0]   = ARRAY(iend-istart,j);
          thisProxy(thisIndex+1).receiveGhost(msg);
      }
    }

    // Receive top rigth grid value and plop in 0,0 position
    void processCorner(cornerMsg *msg) {

      ARRAY(0,0) = msg->gp[0];
      delete msg;
    }

    // send the top right grid value to chare zero
    void sendCorner(void) {

      cornerMsg *msg = new (1) cornerMsg();
      CkSetRefNum(msg, iterations);
      msg->gp[0]   = -ARRAY(iend-istart,n-1);
      thisProxy(0).receiveCorner(msg);
    }

};

#include "p2p.def.h"
