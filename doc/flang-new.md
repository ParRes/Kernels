This works, but -flang-experimental-exec` and `-Wall` are ignored.

```
/opt/llvm/latest/bin/flang-new -flang-experimental-exec -g -O3 -ffast-math -Wall  -DRADIUS=2 -DSTAR -c p2p.F90
ld -L /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib -lSystem p2p.o prk_mod.o -o p2p /opt/llvm/latest/lib/libFortran*a
```
