// declare the kernel name used in SYCL parallel_for
template <typename T> class star1_1d;

template <typename T>
void star1(sycl::queue & q, const size_t n, sycl::buffer<T> & d_in, sycl::buffer<T> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for<class star1_1d<T>>(sycl::range<2> {n-2,n-2}, sycl::id<2> {1,1}, [=] (sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * static_cast<T>(0.5)
                              +in[it[0]*n+(it[1]-1)] * static_cast<T>(-0.5)
                              +in[(it[0]+1)*n+it[1]] * static_cast<T>(0.5)
                              +in[(it[0]-1)*n+it[1]] * static_cast<T>(-0.5);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star1_2d;

template <typename T>
void star1(sycl::queue & q, const size_t n, sycl::buffer<T, 2> & d_in, sycl::buffer<T, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    h.parallel_for<class star1_2d<T>>(sycl::range<2> {n-2,n-2}, sycl::id<2> {1,1}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<T>(0.5)
                   +in[xy-dx1] * static_cast<T>(-0.5)
                   +in[xy+dy1] * static_cast<T>(0.5)
                   +in[xy-dy1] * static_cast<T>(-0.5);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star2_1d;

template <typename T>
void star2(sycl::queue & q, const size_t n, sycl::buffer<T> & d_in, sycl::buffer<T> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for<class star2_1d<T>>(sycl::range<2> {n-4,n-4}, sycl::id<2> {2,2}, [=] (sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * static_cast<T>(0.25)
                              +in[it[0]*n+(it[1]-1)] * static_cast<T>(-0.25)
                              +in[(it[0]+1)*n+it[1]] * static_cast<T>(0.25)
                              +in[(it[0]-1)*n+it[1]] * static_cast<T>(-0.25)
                              +in[it[0]*n+(it[1]+2)] * static_cast<T>(0.125)
                              +in[it[0]*n+(it[1]-2)] * static_cast<T>(-0.125)
                              +in[(it[0]+2)*n+it[1]] * static_cast<T>(0.125)
                              +in[(it[0]-2)*n+it[1]] * static_cast<T>(-0.125);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star2_2d;

template <typename T>
void star2(sycl::queue & q, const size_t n, sycl::buffer<T, 2> & d_in, sycl::buffer<T, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    h.parallel_for<class star2_2d<T>>(sycl::range<2> {n-4,n-4}, sycl::id<2> {2,2}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<T>(0.25)
                   +in[xy-dx1] * static_cast<T>(-0.25)
                   +in[xy+dy1] * static_cast<T>(0.25)
                   +in[xy-dy1] * static_cast<T>(-0.25)
                   +in[xy+dx2] * static_cast<T>(0.125)
                   +in[xy-dx2] * static_cast<T>(-0.125)
                   +in[xy+dy2] * static_cast<T>(0.125)
                   +in[xy-dy2] * static_cast<T>(-0.125);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star3_1d;

template <typename T>
void star3(sycl::queue & q, const size_t n, sycl::buffer<T> & d_in, sycl::buffer<T> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for<class star3_1d<T>>(sycl::range<2> {n-6,n-6}, sycl::id<2> {3,3}, [=] (sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * static_cast<T>(0.16666666666666666)
                              +in[it[0]*n+(it[1]-1)] * static_cast<T>(-0.16666666666666666)
                              +in[(it[0]+1)*n+it[1]] * static_cast<T>(0.16666666666666666)
                              +in[(it[0]-1)*n+it[1]] * static_cast<T>(-0.16666666666666666)
                              +in[it[0]*n+(it[1]+2)] * static_cast<T>(0.08333333333333333)
                              +in[it[0]*n+(it[1]-2)] * static_cast<T>(-0.08333333333333333)
                              +in[(it[0]+2)*n+it[1]] * static_cast<T>(0.08333333333333333)
                              +in[(it[0]-2)*n+it[1]] * static_cast<T>(-0.08333333333333333)
                              +in[it[0]*n+(it[1]+3)] * static_cast<T>(0.05555555555555555)
                              +in[it[0]*n+(it[1]-3)] * static_cast<T>(-0.05555555555555555)
                              +in[(it[0]+3)*n+it[1]] * static_cast<T>(0.05555555555555555)
                              +in[(it[0]-3)*n+it[1]] * static_cast<T>(-0.05555555555555555);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star3_2d;

template <typename T>
void star3(sycl::queue & q, const size_t n, sycl::buffer<T, 2> & d_in, sycl::buffer<T, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    sycl::id<2> dx3(sycl::range<2> {3,0});
    sycl::id<2> dy3(sycl::range<2> {0,3});
    h.parallel_for<class star3_2d<T>>(sycl::range<2> {n-6,n-6}, sycl::id<2> {3,3}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<T>(0.16666666666666666)
                   +in[xy-dx1] * static_cast<T>(-0.16666666666666666)
                   +in[xy+dy1] * static_cast<T>(0.16666666666666666)
                   +in[xy-dy1] * static_cast<T>(-0.16666666666666666)
                   +in[xy+dx2] * static_cast<T>(0.08333333333333333)
                   +in[xy-dx2] * static_cast<T>(-0.08333333333333333)
                   +in[xy+dy2] * static_cast<T>(0.08333333333333333)
                   +in[xy-dy2] * static_cast<T>(-0.08333333333333333)
                   +in[xy+dx3] * static_cast<T>(0.05555555555555555)
                   +in[xy-dx3] * static_cast<T>(-0.05555555555555555)
                   +in[xy+dy3] * static_cast<T>(0.05555555555555555)
                   +in[xy-dy3] * static_cast<T>(-0.05555555555555555);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star4_1d;

template <typename T>
void star4(sycl::queue & q, const size_t n, sycl::buffer<T> & d_in, sycl::buffer<T> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for<class star4_1d<T>>(sycl::range<2> {n-8,n-8}, sycl::id<2> {4,4}, [=] (sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * static_cast<T>(0.125)
                              +in[it[0]*n+(it[1]-1)] * static_cast<T>(-0.125)
                              +in[(it[0]+1)*n+it[1]] * static_cast<T>(0.125)
                              +in[(it[0]-1)*n+it[1]] * static_cast<T>(-0.125)
                              +in[it[0]*n+(it[1]+2)] * static_cast<T>(0.0625)
                              +in[it[0]*n+(it[1]-2)] * static_cast<T>(-0.0625)
                              +in[(it[0]+2)*n+it[1]] * static_cast<T>(0.0625)
                              +in[(it[0]-2)*n+it[1]] * static_cast<T>(-0.0625)
                              +in[it[0]*n+(it[1]+3)] * static_cast<T>(0.041666666666666664)
                              +in[it[0]*n+(it[1]-3)] * static_cast<T>(-0.041666666666666664)
                              +in[(it[0]+3)*n+it[1]] * static_cast<T>(0.041666666666666664)
                              +in[(it[0]-3)*n+it[1]] * static_cast<T>(-0.041666666666666664)
                              +in[it[0]*n+(it[1]+4)] * static_cast<T>(0.03125)
                              +in[it[0]*n+(it[1]-4)] * static_cast<T>(-0.03125)
                              +in[(it[0]+4)*n+it[1]] * static_cast<T>(0.03125)
                              +in[(it[0]-4)*n+it[1]] * static_cast<T>(-0.03125);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star4_2d;

template <typename T>
void star4(sycl::queue & q, const size_t n, sycl::buffer<T, 2> & d_in, sycl::buffer<T, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    sycl::id<2> dx3(sycl::range<2> {3,0});
    sycl::id<2> dy3(sycl::range<2> {0,3});
    sycl::id<2> dx4(sycl::range<2> {4,0});
    sycl::id<2> dy4(sycl::range<2> {0,4});
    h.parallel_for<class star4_2d<T>>(sycl::range<2> {n-8,n-8}, sycl::id<2> {4,4}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<T>(0.125)
                   +in[xy-dx1] * static_cast<T>(-0.125)
                   +in[xy+dy1] * static_cast<T>(0.125)
                   +in[xy-dy1] * static_cast<T>(-0.125)
                   +in[xy+dx2] * static_cast<T>(0.0625)
                   +in[xy-dx2] * static_cast<T>(-0.0625)
                   +in[xy+dy2] * static_cast<T>(0.0625)
                   +in[xy-dy2] * static_cast<T>(-0.0625)
                   +in[xy+dx3] * static_cast<T>(0.041666666666666664)
                   +in[xy-dx3] * static_cast<T>(-0.041666666666666664)
                   +in[xy+dy3] * static_cast<T>(0.041666666666666664)
                   +in[xy-dy3] * static_cast<T>(-0.041666666666666664)
                   +in[xy+dx4] * static_cast<T>(0.03125)
                   +in[xy-dx4] * static_cast<T>(-0.03125)
                   +in[xy+dy4] * static_cast<T>(0.03125)
                   +in[xy-dy4] * static_cast<T>(-0.03125);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star5_1d;

template <typename T>
void star5(sycl::queue & q, const size_t n, sycl::buffer<T> & d_in, sycl::buffer<T> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    h.parallel_for<class star5_1d<T>>(sycl::range<2> {n-10,n-10}, sycl::id<2> {5,5}, [=] (sycl::item<2> it) {
        out[it[0]*n+it[1]] += +in[it[0]*n+(it[1]+1)] * static_cast<T>(0.1)
                              +in[it[0]*n+(it[1]-1)] * static_cast<T>(-0.1)
                              +in[(it[0]+1)*n+it[1]] * static_cast<T>(0.1)
                              +in[(it[0]-1)*n+it[1]] * static_cast<T>(-0.1)
                              +in[it[0]*n+(it[1]+2)] * static_cast<T>(0.05)
                              +in[it[0]*n+(it[1]-2)] * static_cast<T>(-0.05)
                              +in[(it[0]+2)*n+it[1]] * static_cast<T>(0.05)
                              +in[(it[0]-2)*n+it[1]] * static_cast<T>(-0.05)
                              +in[it[0]*n+(it[1]+3)] * static_cast<T>(0.03333333333333333)
                              +in[it[0]*n+(it[1]-3)] * static_cast<T>(-0.03333333333333333)
                              +in[(it[0]+3)*n+it[1]] * static_cast<T>(0.03333333333333333)
                              +in[(it[0]-3)*n+it[1]] * static_cast<T>(-0.03333333333333333)
                              +in[it[0]*n+(it[1]+4)] * static_cast<T>(0.025)
                              +in[it[0]*n+(it[1]-4)] * static_cast<T>(-0.025)
                              +in[(it[0]+4)*n+it[1]] * static_cast<T>(0.025)
                              +in[(it[0]-4)*n+it[1]] * static_cast<T>(-0.025)
                              +in[it[0]*n+(it[1]+5)] * static_cast<T>(0.02)
                              +in[it[0]*n+(it[1]-5)] * static_cast<T>(-0.02)
                              +in[(it[0]+5)*n+it[1]] * static_cast<T>(0.02)
                              +in[(it[0]-5)*n+it[1]] * static_cast<T>(-0.02);
    });
  });
}

// declare the kernel name used in SYCL parallel_for
template <typename T> class star5_2d;

template <typename T>
void star5(sycl::queue & q, const size_t n, sycl::buffer<T, 2> & d_in, sycl::buffer<T, 2> & d_out)
{
  q.submit([&](sycl::handler& h) {
    auto in  = d_in.template get_access<sycl::access::mode::read>(h);
    auto out = d_out.template get_access<sycl::access::mode::read_write>(h);
    sycl::id<2> dx1(sycl::range<2> {1,0});
    sycl::id<2> dy1(sycl::range<2> {0,1});
    sycl::id<2> dx2(sycl::range<2> {2,0});
    sycl::id<2> dy2(sycl::range<2> {0,2});
    sycl::id<2> dx3(sycl::range<2> {3,0});
    sycl::id<2> dy3(sycl::range<2> {0,3});
    sycl::id<2> dx4(sycl::range<2> {4,0});
    sycl::id<2> dy4(sycl::range<2> {0,4});
    sycl::id<2> dx5(sycl::range<2> {5,0});
    sycl::id<2> dy5(sycl::range<2> {0,5});
    h.parallel_for<class star5_2d<T>>(sycl::range<2> {n-10,n-10}, sycl::id<2> {5,5}, [=] (sycl::item<2> it) {
        sycl::id<2> xy = it.get_id();
        out[xy] += +in[xy+dx1] * static_cast<T>(0.1)
                   +in[xy-dx1] * static_cast<T>(-0.1)
                   +in[xy+dy1] * static_cast<T>(0.1)
                   +in[xy-dy1] * static_cast<T>(-0.1)
                   +in[xy+dx2] * static_cast<T>(0.05)
                   +in[xy-dx2] * static_cast<T>(-0.05)
                   +in[xy+dy2] * static_cast<T>(0.05)
                   +in[xy-dy2] * static_cast<T>(-0.05)
                   +in[xy+dx3] * static_cast<T>(0.03333333333333333)
                   +in[xy-dx3] * static_cast<T>(-0.03333333333333333)
                   +in[xy+dy3] * static_cast<T>(0.03333333333333333)
                   +in[xy-dy3] * static_cast<T>(-0.03333333333333333)
                   +in[xy+dx4] * static_cast<T>(0.025)
                   +in[xy-dx4] * static_cast<T>(-0.025)
                   +in[xy+dy4] * static_cast<T>(0.025)
                   +in[xy-dy4] * static_cast<T>(-0.025)
                   +in[xy+dx5] * static_cast<T>(0.02)
                   +in[xy-dx5] * static_cast<T>(-0.02)
                   +in[xy+dy5] * static_cast<T>(0.02)
                   +in[xy-dy5] * static_cast<T>(-0.02);
    });
  });
}

