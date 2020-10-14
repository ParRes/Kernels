package main

import (
    "fmt"
    "flag"
    "os"
)

func main() {

  fmt.Println("Parallel Research Kernels")
  fmt.Println("Go STREAM triad: A = B + scalar * C")

  iterations := flag.Int("i", 0, "iterations")
  length     := flag.Int64("n", 0, "length of vector")

  flag.Parse()

  if len(os.Args) < 2 {
    fmt.Println("Usage: <# iterations> <vector length>")
    os.Exit(1)
  }

  if (iterations < 1) {
    fmt.Println("ERROR: iterations must be >= 1")
    os.Exit(1)
  }

  if (length <= 0) {
    fmt.Println("ERROR: vector length must be positive")
    os.Exit(1)
  }

  fmt.Println("Number of iterations = ", iterations)
  fmt.Println("Vector length        = ", length)

}
