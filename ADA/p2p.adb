with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Real_Time,
    Ada.Command_line;

procedure p2p is

    use
        Ada.Text_IO,
        Ada.Integer_Text_IO,
        Ada.Real_Time,
        Ada.Command_line;

    -- GNAT Integer = int32_t, Long_Integer = int64_t
    Iterations : Natural := 10;
    M : Natural := 1_000;
    N : Natural := 1_000;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial pipeline execution on 2D grid");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        M := Integer'Value(Argument(2));
    end if;
    if Argument_Count > 2 then
        N := Integer'Value(Argument(3));
    end if;

    if Iterations < 1 then
        Put_Line("Iteration count must be greater than " & Integer'Image(Iterations) );
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Grid sizes            =" & Integer'Image(M) & " *" & Integer'Image(N) );

    declare

        -- Double precision throughout, matching the "double precision"
        -- convention of the other language ports of this kernel:
        -- Ada.Numerics.Real_Arrays.Real_Matrix is single-precision Float,
        -- which is not accurate enough for the corner-value recurrence
        -- below once M/N (and thus the number of accumulated terms) grow.
        -- Heap-allocated (like nstream_array.adb), not a plain local
        -- array: an M-by-N Long_Float grid at realistic benchmark sizes
        -- (e.g. 1000x1000 = 8MB) does not reliably fit on the default
        -- 8MB stack.
        type Grid_Matrix is array(Integer range <>, Integer range <>) of Long_Float;
        type Grid_Matrix_Access is access Grid_Matrix;
        Grid : Grid_Matrix_Access := new Grid_Matrix(1..M,1..N);

        T0, T1 : Time;
        DT : Time_Span;

        US : constant Time_Span := Microseconds(US => Iterations);

        K : Integer := 0;

        CornerVal : Long_Float;
        Epsilon : constant Long_Float := 1.0E-8;

        AvgTime : Long_Float;
        Nstream_us : Integer;

    begin

-- initialization: left column and top row carry a boundary ramp, everything
-- else starts at zero, same as the other language ports' p2p kernel.

        for J in 1..N Loop
            Grid(1,J) := Long_Float(J-1);
        end Loop;
        for I in 1..M Loop
            Grid(I,1) := Long_Float(I-1);
        end Loop;
        for I in 2..M Loop
            for J in 2..N Loop
                Grid(I,J) := 0.0;
            end Loop;
        end Loop;

-- run the experiment

        for K in 0..Iterations Loop

            if K = 1 then
                T0 := Clock;
            end if;

            for I in 2..M Loop
                for J in 2..N Loop
                    Grid(I,J) := Grid(I-1,J) + Grid(I,J-1) - Grid(I-1,J-1);
                end Loop;
            end Loop;

            -- copy top right corner value to bottom left corner to create
            -- a dependency between successive iterations
            Grid(1,1) := -Grid(M,N);

        end Loop;
        T1 := Clock;
        DT := T1 - T0;

-- validation

        CornerVal := Long_Float((Iterations+1) * (M+N-2));

        if abs ( Grid(M,N) - CornerVal ) / CornerVal > Epsilon then
            Put_Line("ERROR: checksum " & Long_Float'Image(Grid(M,N)) & " does not match verification value " & Long_Float'Image(CornerVal) );
        else
            Put_Line("Solution validates");
            Nstream_us := DT / US; -- average time per iteration, in microseconds, thanks to US
            AvgTime := Long_Float(Nstream_us) / Long_Float(1_000_000);
            Put_Line("Avg time (s): " & Long_Float'Image(AvgTime) );
            Put_Line("Rate (MFlops/s): " & Long_Float'Image(1.0E-6 * 2.0 * Long_Float((M-1)*(N-1)) / AvgTime) );
        end if;

    end;

end p2p;
