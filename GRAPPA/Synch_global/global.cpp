/*
Copyright (c) 2013, Intel Corporation
Copyright (c) 2015, John Abercrombie

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
* Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    StopNGo

PURPOSE: This program tests the efficiency of a global synchronization
         on the target system. 
  
USAGE:   The program takes as input the number of times the test of
         string manipulation involving the global synchronization is
         carried out, as well as the length of the string.

         <progname>  <# iterations> <length of numerical string>
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

         Compile with VERBOSE defined if you want lots of output.

FUNCTIONS CALLED:

         Other than MPI or standard C functions, the following 
         functions are used in this program:

         chartoi()

HISTORY: Written by Rob Van der Wijngaart, December 2005.
         Adapted for Grappa              , September 2015.
  
*******************************************************************/

#include <stdint.h>
#include <par-res-kern_general.h>
#include <Grappa.hpp>

using namespace Grappa;

#define EOS '\0'
#define root 0

GlobalCompletionEvent gce;

int chartoi (char c) {
  /* define short string; need two characters, second contains string terminator */
  char letter[2] = "0";
  letter[0] = c;
  return (atoi(letter));
}

int main(int argc, char *argv[]) {
  Grappa::init(&argc, &argv);

  int Num_procs = Grappa::cores();

  if (argc != 3) {
    if (Grappa::mycore() == root)
      std::cout<<"Usage: "<<argv[0]<<" <#iterations> <scramble string length>"<<std::endl;
    exit(1);
  }

  int iterations = atoi(argv[1]);
  if (iterations < 1) {
    if (Grappa::mycore() == root)
      std::cout<<"ERROR: iterations must be positive: "<<iterations<<std::endl;
    exit(1);
  }

  int64_t length = atol(argv[2]);
  if (length < Num_procs || length%Num_procs != 0) {
    if (Grappa::mycore() == root)
      std::cout<<"ERROR: length of string "<<length<<" must be multiple of # cores: "<<Num_procs;
    exit(1);
  }

  Grappa::run( [Num_procs,iterations,length] {
      static int my_ID;
      int64_t proc_length = length/Num_procs;
      GlobalAddress<char> basestring, catstring;
      static char * iterstring;
      char const * scramble = "27638472638746283742712311207892";
      double stopngo_time;

      basestring = global_alloc<char>(proc_length+1);
      /* fill the base string with copies (truncated) of scrable string         */
      forall( basestring, proc_length, [scramble] (int64_t index, char& c) {
	  c = scramble[index%32];
	});
      Grappa::delegate::write(basestring+length, EOS);
      LOG(INFO)<<"basestring = "<<(char *)basestring.pointer();

      catstring = global_alloc<char>(length+1);
      /* initialize concatenation string with nonsense                          */
      Grappa::memset(catstring, '9', length);
      Grappa::delegate::write(catstring+length, EOS);

      Grappa::on_all_cores([=] {
	  my_ID = mycore();
	  
	  /* everybody recieves a private copy of the base string                   */
	  iterstring = new char[proc_length+1];
	  if (!iterstring) {
	    std::cout<<"ERROR: Could not allocate space for strings in rank"<<my_ID<<std::endl;
	    exit(1);
	  }
	  for (int i = 0; i <= proc_length; i++)
	    iterstring[i] = Grappa::delegate::read(basestring+i);

	  //	  LOG(INFO)<<"Core "<<my_ID<<"'s iterstring = "<<iterstring;
	});

      std::cout<<"Grappa global syncronization test"<<std::endl;
      std::cout<<"Number of threads         = "<<Num_procs<<std::endl;
      std::cout<<"Length of scramble string = "<<length<<std::endl;
      std::cout<<"Number of iterations      = "<<iterations<<std::endl;

      // execute kernel
      stopngo_time = Grappa::walltime();

      auto completion_target = local_gce.enroll_recurring(cores());

      Grappa::on_all_cores( [=] {
	  // iteration loop
	  for (int iter = 0; iter <iterations; iter++) {

	    // glue local string into global synch string
	    for (int i = 0; i < proc_length; i++) {
	      auto message_completion_target = local_gce.enroll();
	      char val = iterstring[i];
	      delegate::call<async>(catstring+my_ID*proc_length+i, [val,message_completion_target] (char& c) {
		  c = val;
		  local_gce.complete(message_completion_target);
		});
	    }

	    local_gce.complete( completion_target );
	    local_gce.wait();

	    // each core receives a different substring of the global catstring
	    for (int i = 0; i < proc_length; i++) {
	      auto message_completion_target = local_gce.enroll();
	      iterstring[i] = delegate::call(catstring+my_ID+i*Num_procs, [message_completion_target](char& c) {
		  local_gce.complete(message_completion_target);
		  return c; });
	    }

	    local_gce.complete( completion_target );
	    local_gce.wait();
	  }
	});

      stopngo_time = Grappa::walltime() - stopngo_time;

      LOG(INFO)<<"Kernel complete";

      /* compute checksum on obtained result, adding all digits in the string   */
      int64_t basesum=0;
      for (int i=0; i<proc_length; i++) 
	basesum += delegate::call(basestring+i,[](char& c)->int {return chartoi(c);});

      int64_t checksum=0;
      for (int i=0; i<length;i++)
	checksum += delegate::call(catstring+i,[](char& c)->int {return chartoi(c);});
      if (checksum != basesum*Num_procs) {
	std::cout<<"Incorrect checksum: "<<checksum<<" instead of "<<basesum*Num_procs<<std::endl;
	exit(1);
      }
      else {
	std::cout<<"Solution Validates"<<std::endl;
	std::cout<<"Rate (synch/s): "<<iterations/stopngo_time
		 <<", time (s): "<<stopngo_time<<std::endl;
      }
    });


  Grappa::finalize();
}
