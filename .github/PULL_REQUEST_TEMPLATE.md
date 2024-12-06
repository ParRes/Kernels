If this pull request is fixing a bug, please link the associated issue.
The rest of this template does not apply.

If this pull request is providing a new implementation of the PRKs,
please use the following template.

Note that checking all of the boxes is not required.

## New PRK implementation checklist

### Which kernels are implemented?

- [ ] synch_p2p (p2p)
- [ ] stencil
- [ ] transpose
- [ ] nstream
- [ ] dgemm
- [ ] reduce
- [ ] sparse
- [ ] branch
- [ ] random
- [ ] refcount
- [ ] synch_global
- [ ] PIC
- [ ] AMR

### Documentation and build examples

If your implementation uses a new programming model that is not
ubiquitious (i.e. included in the system compiler on most systems)
then you need to provide a link to the appropriate documentation
for a new user to install it, etc.

We strongly recommend that you add the appropriate features
to `make.defs.${toolchain}` if appropriate.

### Do you certify that your contribution is made in good faith and does not attempt to introduce any negative behavior into this project?

- [ ] Yes
- [ ] No
