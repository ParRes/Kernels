println("ARGS=",ARGS)

int_args = map(x->parse(Int64,x),ARGS)

println("int_args=",int_args)

for x in ARGS
    println(x,",",typeof(x))
end

i = convert(Int,ARGS)
