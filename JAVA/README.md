# How to build

Just type `make`

# How to run

You have to set the classpath to the current directory, at least some of the time.

```
java -cp . nstream 10 10000000
java -cp . stencil 10 1000
java -cp . stencil 10 1000 star 4
java -cp . stencil 10 1000 grid 2
java -cp . transpose 10 1000
java -cp . transpose 10 1000 32
```
