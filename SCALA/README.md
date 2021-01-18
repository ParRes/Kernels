# How to run

Scala programs can be run directly as scripts.
```
./nstream.scala  10 $((1024*1024*64))
```
This may not be optimal but I am a novice and did not figure out
how to compile Scala to Java yet.

Note that the first line of each program sets the maximum memory
used by Java to 4G, which is probably acceptable for most use cases.
```
#!/usr/bin/env -S scala -nc -J-Xmx4g
```
The default is quite low and will not allow you to run nstream
with more than ~16MW.
