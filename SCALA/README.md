# How to build

Just type `make`

# How to run

```
JAVA_OPTS="-Xmx4G" scala nstream 10 $((1024*1024*64))
```

Note that the environmental variable JAVA_OPTS sets the maximum memory
used by Java to 4G, which is probably acceptable for most use cases.
The default is quite low and will not allow you to run nstream with
more than ~16MW.

If you're interested in running in a script mode, simple specify the
file name of the source code.

```
JAVA_OPTS="-Xmx4G" scala nstream.scala 10 $((1024*1024*64))
```

