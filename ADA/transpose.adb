with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Real_Time,
    Ada.Command_line;

procedure transpose is

    use
        Ada.Text_IO,
        Ada.Integer_Text_IO,
        Ada.Real_Time,
        Ada.Command_line;

    -- GNAT Integer = int32_t, Long_Integer = int64_t
    Iterations : Natural := 10;
    Order : Natural := 1_000;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial Matrix transpose: B += A^T");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        Order := Integer'Value(Argument(2));
    end if;

    if Iterations < 1 then
        Put_Line("Iteration count must be greater than " & Integer'Image(Iterations) );
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Matrix order         =" & Integer'Image(Order) );

    declare

        -- Double precision throughout: Ada.Numerics.Real_Arrays.Real_Matrix
        -- is single-precision Float, not accurate enough to validate at
        -- 1.0E-8 once Order (and the running sums in the closed-form
        -- check below) get large. Heap-allocated (like nstream_array.adb):
        -- two Order-by-Order Long_Float matrices at realistic benchmark
        -- sizes (e.g. 1000x1000 = 8MB each) do not reliably fit on the
        -- default 8MB stack.
        type Real_Matrix is array(Integer range <>, Integer range <>) of Long_Float;
        type Real_Matrix_Access is access Real_Matrix;
        A : Real_Matrix_Access := new Real_Matrix(1..Order,1..Order);
        B : Real_Matrix_Access := new Real_Matrix(1..Order,1..Order);

        T0, T1 : Time;
        DT : Time_Span;

        US : constant Time_Span := Microseconds(US => Iterations);

        K : Integer := 0;
        AbsErr : Long_Float := 0.0;
        Addit : Long_Float;

        AvgTime : Long_Float;
        Bytes : Long_Integer;
        Nstream_us : Integer;
        Epsilon : constant Long_Float := 1.0E-8;

    begin

-- initialization: A(i,j) = i*order + j (0-based), same convention as the
-- other language ports of this kernel

        for I in 1..Order Loop
            for J in 1..Order Loop
                A(I,J) := Long_Float((I-1)*Order + (J-1));
                B(I,J) := 0.0;
            end Loop;
        end Loop;

-- run the experiment

        for K in 0..Iterations Loop

            if K = 1 then
                T0 := Clock;
            end if;

            -- transpose the matrix: B += A^T, then bump A so successive
            -- iterations touch fresh data (matches every other language's
            -- transpose kernel)
            for I in 1..Order Loop
                for J in 1..Order Loop
                    B(J,I) := B(J,I) + A(I,J);
                    A(I,J) := A(I,J) + 1.0;
                end Loop;
            end Loop;

        end Loop;
        T1 := Clock;
        DT := T1 - T0;

-- validation: every B(j,i) should equal the closed-form sum of the
-- (Iterations+1) transposed-and-incremented values of A(i,j)

        Addit := (Long_Float(Iterations+1) * Long_Float(Iterations)) / 2.0;
        for J in 1..Order Loop
            for I in 1..Order Loop
                AbsErr := AbsErr + abs ( B(J,I)
                    - ( Long_Float((Order*(I-1) + (J-1))) * Long_Float(Iterations+1) + Addit ) );
            end Loop;
        end Loop;

        if AbsErr > Epsilon then
            Put_Line("ERROR: Aggregate squared error " & Long_Float'Image(AbsErr) & " exceeds threshold " & Long_Float'Image(Epsilon) );
        else
            Put_Line("Solution validates");
            Bytes := 2 * Long_Integer(Long_Float'Size / 8) * Long_Integer(Order) * Long_Integer(Order);
            Nstream_us := DT / US; -- this is per iteration now, thanks to US
            AvgTime := Long_Float(Nstream_us) / Long_Float(1_000_000);
            Put_Line("Avg time (s): " & Long_Float'Image(AvgTime) );
            Put_Line("Rate (MB/s): " & Long_Float'Image(Long_Float(Bytes) / Long_Float(Nstream_us)) );
        end if;

    end;

end transpose;
