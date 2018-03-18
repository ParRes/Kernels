void star1(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double> & d_in,
           cl::sycl::buffer<double> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star1_1d>(cl::sycl::range<2> {n-2*1,n-2*1}, cl::sycl::id<2> {1,1},
                                [=] (cl::sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * 0.5
                              +in[it[0]*n+(it[1]-1)] * -0.5
                              +in[(it[0]+1)*n+it[1]] * 0.5
                              +in[(it[0]-1)*n+it[1]] * -0.5;
    });
  });
}

void star1(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double, 2> & d_in,
           cl::sycl::buffer<double, 2> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star1_2d>(cl::sycl::range<2> {n-2*1,n-2*1}, cl::sycl::id<2> {1,1},
                                [=] (cl::sycl::item<2> it) {
        cl::sycl::id<2> xy = it.get_id();
        cl::sycl::id<2> dx1(cl::sycl::range<2> {1,0});
        cl::sycl::id<2> dy1(cl::sycl::range<2> {0,1});
        out[xy] += +in[xy+dx1] * 0.5
                   +in[xy-dx1] * -0.5
                   +in[xy+dy1] * 0.5
                   +in[xy-dy1] * -0.5;
    });
  });
}

void star2(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double> & d_in,
           cl::sycl::buffer<double> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star2_1d>(cl::sycl::range<2> {n-2*2,n-2*2}, cl::sycl::id<2> {2,2},
                                [=] (cl::sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * 0.25
                              +in[it[0]*n+(it[1]-1)] * -0.25
                              +in[(it[0]+1)*n+it[1]] * 0.25
                              +in[(it[0]-1)*n+it[1]] * -0.25
                              +in[it[0]*n+(it[1]+2)] * 0.125
                              +in[it[0]*n+(it[1]-2)] * -0.125
                              +in[(it[0]+2)*n+it[1]] * 0.125
                              +in[(it[0]-2)*n+it[1]] * -0.125;
    });
  });
}

void star2(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double, 2> & d_in,
           cl::sycl::buffer<double, 2> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star2_2d>(cl::sycl::range<2> {n-2*2,n-2*2}, cl::sycl::id<2> {2,2},
                                [=] (cl::sycl::item<2> it) {
        cl::sycl::id<2> xy = it.get_id();
        cl::sycl::id<2> dx1(cl::sycl::range<2> {1,0});
        cl::sycl::id<2> dy1(cl::sycl::range<2> {0,1});
        cl::sycl::id<2> dx2(cl::sycl::range<2> {2,0});
        cl::sycl::id<2> dy2(cl::sycl::range<2> {0,2});
        out[xy] += +in[xy+dx1] * 0.25
                   +in[xy-dx1] * -0.25
                   +in[xy+dy1] * 0.25
                   +in[xy-dy1] * -0.25
                   +in[xy+dx2] * 0.125
                   +in[xy-dx2] * -0.125
                   +in[xy+dy2] * 0.125
                   +in[xy-dy2] * -0.125;
    });
  });
}

void star3(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double> & d_in,
           cl::sycl::buffer<double> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star3_1d>(cl::sycl::range<2> {n-2*3,n-2*3}, cl::sycl::id<2> {3,3},
                                [=] (cl::sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * 0.166666666667
                              +in[it[0]*n+(it[1]-1)] * -0.166666666667
                              +in[(it[0]+1)*n+it[1]] * 0.166666666667
                              +in[(it[0]-1)*n+it[1]] * -0.166666666667
                              +in[it[0]*n+(it[1]+2)] * 0.0833333333333
                              +in[it[0]*n+(it[1]-2)] * -0.0833333333333
                              +in[(it[0]+2)*n+it[1]] * 0.0833333333333
                              +in[(it[0]-2)*n+it[1]] * -0.0833333333333
                              +in[it[0]*n+(it[1]+3)] * 0.0555555555556
                              +in[it[0]*n+(it[1]-3)] * -0.0555555555556
                              +in[(it[0]+3)*n+it[1]] * 0.0555555555556
                              +in[(it[0]-3)*n+it[1]] * -0.0555555555556;
    });
  });
}

void star3(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double, 2> & d_in,
           cl::sycl::buffer<double, 2> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star3_2d>(cl::sycl::range<2> {n-2*3,n-2*3}, cl::sycl::id<2> {3,3},
                                [=] (cl::sycl::item<2> it) {
        cl::sycl::id<2> xy = it.get_id();
        cl::sycl::id<2> dx1(cl::sycl::range<2> {1,0});
        cl::sycl::id<2> dy1(cl::sycl::range<2> {0,1});
        cl::sycl::id<2> dx2(cl::sycl::range<2> {2,0});
        cl::sycl::id<2> dy2(cl::sycl::range<2> {0,2});
        cl::sycl::id<2> dx3(cl::sycl::range<2> {3,0});
        cl::sycl::id<2> dy3(cl::sycl::range<2> {0,3});
        out[xy] += +in[xy+dx1] * 0.166666666667
                   +in[xy-dx1] * -0.166666666667
                   +in[xy+dy1] * 0.166666666667
                   +in[xy-dy1] * -0.166666666667
                   +in[xy+dx2] * 0.0833333333333
                   +in[xy-dx2] * -0.0833333333333
                   +in[xy+dy2] * 0.0833333333333
                   +in[xy-dy2] * -0.0833333333333
                   +in[xy+dx3] * 0.0555555555556
                   +in[xy-dx3] * -0.0555555555556
                   +in[xy+dy3] * 0.0555555555556
                   +in[xy-dy3] * -0.0555555555556;
    });
  });
}

void star4(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double> & d_in,
           cl::sycl::buffer<double> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star4_1d>(cl::sycl::range<2> {n-2*4,n-2*4}, cl::sycl::id<2> {4,4},
                                [=] (cl::sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * 0.125
                              +in[it[0]*n+(it[1]-1)] * -0.125
                              +in[(it[0]+1)*n+it[1]] * 0.125
                              +in[(it[0]-1)*n+it[1]] * -0.125
                              +in[it[0]*n+(it[1]+2)] * 0.0625
                              +in[it[0]*n+(it[1]-2)] * -0.0625
                              +in[(it[0]+2)*n+it[1]] * 0.0625
                              +in[(it[0]-2)*n+it[1]] * -0.0625
                              +in[it[0]*n+(it[1]+3)] * 0.0416666666667
                              +in[it[0]*n+(it[1]-3)] * -0.0416666666667
                              +in[(it[0]+3)*n+it[1]] * 0.0416666666667
                              +in[(it[0]-3)*n+it[1]] * -0.0416666666667
                              +in[it[0]*n+(it[1]+4)] * 0.03125
                              +in[it[0]*n+(it[1]-4)] * -0.03125
                              +in[(it[0]+4)*n+it[1]] * 0.03125
                              +in[(it[0]-4)*n+it[1]] * -0.03125;
    });
  });
}

void star4(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double, 2> & d_in,
           cl::sycl::buffer<double, 2> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star4_2d>(cl::sycl::range<2> {n-2*4,n-2*4}, cl::sycl::id<2> {4,4},
                                [=] (cl::sycl::item<2> it) {
        cl::sycl::id<2> xy = it.get_id();
        cl::sycl::id<2> dx1(cl::sycl::range<2> {1,0});
        cl::sycl::id<2> dy1(cl::sycl::range<2> {0,1});
        cl::sycl::id<2> dx2(cl::sycl::range<2> {2,0});
        cl::sycl::id<2> dy2(cl::sycl::range<2> {0,2});
        cl::sycl::id<2> dx3(cl::sycl::range<2> {3,0});
        cl::sycl::id<2> dy3(cl::sycl::range<2> {0,3});
        cl::sycl::id<2> dx4(cl::sycl::range<2> {4,0});
        cl::sycl::id<2> dy4(cl::sycl::range<2> {0,4});
        out[xy] += +in[xy+dx1] * 0.125
                   +in[xy-dx1] * -0.125
                   +in[xy+dy1] * 0.125
                   +in[xy-dy1] * -0.125
                   +in[xy+dx2] * 0.0625
                   +in[xy-dx2] * -0.0625
                   +in[xy+dy2] * 0.0625
                   +in[xy-dy2] * -0.0625
                   +in[xy+dx3] * 0.0416666666667
                   +in[xy-dx3] * -0.0416666666667
                   +in[xy+dy3] * 0.0416666666667
                   +in[xy-dy3] * -0.0416666666667
                   +in[xy+dx4] * 0.03125
                   +in[xy-dx4] * -0.03125
                   +in[xy+dy4] * 0.03125
                   +in[xy-dy4] * -0.03125;
    });
  });
}

void star5(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double> & d_in,
           cl::sycl::buffer<double> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star5_1d>(cl::sycl::range<2> {n-2*5,n-2*5}, cl::sycl::id<2> {5,5},
                                [=] (cl::sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * 0.1
                              +in[it[0]*n+(it[1]-1)] * -0.1
                              +in[(it[0]+1)*n+it[1]] * 0.1
                              +in[(it[0]-1)*n+it[1]] * -0.1
                              +in[it[0]*n+(it[1]+2)] * 0.05
                              +in[it[0]*n+(it[1]-2)] * -0.05
                              +in[(it[0]+2)*n+it[1]] * 0.05
                              +in[(it[0]-2)*n+it[1]] * -0.05
                              +in[it[0]*n+(it[1]+3)] * 0.0333333333333
                              +in[it[0]*n+(it[1]-3)] * -0.0333333333333
                              +in[(it[0]+3)*n+it[1]] * 0.0333333333333
                              +in[(it[0]-3)*n+it[1]] * -0.0333333333333
                              +in[it[0]*n+(it[1]+4)] * 0.025
                              +in[it[0]*n+(it[1]-4)] * -0.025
                              +in[(it[0]+4)*n+it[1]] * 0.025
                              +in[(it[0]-4)*n+it[1]] * -0.025
                              +in[it[0]*n+(it[1]+5)] * 0.02
                              +in[it[0]*n+(it[1]-5)] * -0.02
                              +in[(it[0]+5)*n+it[1]] * 0.02
                              +in[(it[0]-5)*n+it[1]] * -0.02;
    });
  });
}

void star5(cl::sycl::queue & q, const size_t n,
           cl::sycl::buffer<double, 2> & d_in,
           cl::sycl::buffer<double, 2> & d_out) {
  q.submit([&](cl::sycl::handler& h) {
    auto in  = d_in.get_access<cl::sycl::access::mode::read>(h);
    auto out = d_out.get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<class star5_2d>(cl::sycl::range<2> {n-2*5,n-2*5}, cl::sycl::id<2> {5,5},
                                [=] (cl::sycl::item<2> it) {
        cl::sycl::id<2> xy = it.get_id();
        cl::sycl::id<2> dx1(cl::sycl::range<2> {1,0});
        cl::sycl::id<2> dy1(cl::sycl::range<2> {0,1});
        cl::sycl::id<2> dx2(cl::sycl::range<2> {2,0});
        cl::sycl::id<2> dy2(cl::sycl::range<2> {0,2});
        cl::sycl::id<2> dx3(cl::sycl::range<2> {3,0});
        cl::sycl::id<2> dy3(cl::sycl::range<2> {0,3});
        cl::sycl::id<2> dx4(cl::sycl::range<2> {4,0});
        cl::sycl::id<2> dy4(cl::sycl::range<2> {0,4});
        cl::sycl::id<2> dx5(cl::sycl::range<2> {5,0});
        cl::sycl::id<2> dy5(cl::sycl::range<2> {0,5});
        out[xy] += +in[xy+dx1] * 0.1
                   +in[xy-dx1] * -0.1
                   +in[xy+dy1] * 0.1
                   +in[xy-dy1] * -0.1
                   +in[xy+dx2] * 0.05
                   +in[xy-dx2] * -0.05
                   +in[xy+dy2] * 0.05
                   +in[xy-dy2] * -0.05
                   +in[xy+dx3] * 0.0333333333333
                   +in[xy-dx3] * -0.0333333333333
                   +in[xy+dy3] * 0.0333333333333
                   +in[xy-dy3] * -0.0333333333333
                   +in[xy+dx4] * 0.025
                   +in[xy-dx4] * -0.025
                   +in[xy+dy4] * 0.025
                   +in[xy-dy4] * -0.025
                   +in[xy+dx5] * 0.02
                   +in[xy-dx5] * -0.02
                   +in[xy+dy5] * 0.02
                   +in[xy-dy5] * -0.02;
    });
  });
}

