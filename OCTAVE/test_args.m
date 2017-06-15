#!/usr/bin/env octave -qf

printf("program name: %s\n", program_name());
arg_list = argv();
for i = 1:nargin
    printf("argument{%d}: %s\n", i, arg_list{i});
end
printf("\n");
