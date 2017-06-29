# Who can contribute

The Parallel Research Kernels (PRK) project is an open research effort that encourages contributions
from anyone in the community with relevant skills.  *We value contributions regardless of the
contributor's age, race, gender or gender identity, sexual orientation, national origin, religion,
educational experience, or employment status.*

We especially encourage contributions from researchers with unique skills that the core team does
not have.  If you are an expert in a programming model that is not represented in the PRK suite,
your contributions will be much appreciated.  We also appreciate minor contributions, although 
we prefer you not spend your time on whitespace cleanup.

# How to contribute

We encourage all contributors to fork this project on GitHub and follow the model described 
[here](https://guides.github.com/introduction/flow/).

The PRK projects relies on [Travis CI](https://travis-ci.org/) to catch scale-independent bugs,
which is to say, most bugs can be caught here, but some bugs will only appear at scale or when
tested in a distributed memory environment.  Passing Travis testing is a necessary and often
sufficient criteria for merging pull requests.  If your contributions won't be tested by Travis,
we strongly encourage you to make the necessary changes to support that, or ask us for help
doing so.

# Copyright and licensing

The PRK project uses the [3-clause BSD](https://opensource.org/licenses/BSD-3-Clause) (BSD-3) license, 
with some additional conditions inherited from STREAM.
See [COPYING](https://github.com/ParRes/Kernels/blob/master/COPYING) for details.

All contributions to the PRK project inherit the BSD-3 license unless a contributor explicitly 
states otherwise.  If your contributions require a different license, you must note this
explicitly in your pull request, and some license changes will prevent us from being able to
accept your contribution.  For example, no form of GPL can be used, because prevents the use
of the code for research that requires confidentiality.
See [this](https://www.gnu.org/licenses/gpl-faq.en.html#DoesTheGPLAllowNDA) for details.
