with
    Ada.Numerics.Real_Arrays,
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Real_Time,
    Ada.Command_line;

procedure transpose is

    use
        Ada.Numerics.Real_Arrays,
        Ada.Text_IO,
        Ada.Integer_Text_IO,
        Ada.Real_Time,
        Ada.Command_line;

    -- GNAT Integer = int32_t, Long_Integer = int64_t
    Iterations : Natural := 10;
    Order : Natural := 1_000;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial Transpose B += A^T");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        Order := Integer'Value(Argument(2));
    end if;

    if Iterations < 2 then
        Put_Line("Iteration count must be greater than " & Integer'Image(Iterations) );
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Matrix order         =" & Integer'Image(Order) );

    declare

        I : Natural := 1;
        J : Natural := 1;
        A : Real_Matrix(1..Order,1..Order);
        B : Real_Matrix(1..Order,1..Order);

        T0, T1 : Time;
        DT : Time_Span;

        US : constant Time_Span := Microseconds(US => Iterations);

        K : Integer := 0;
        Asum : Float := 0.0;

        AvgTime : Duration;
        Bytes : Long_Integer;
        Nstream_us : Integer;
        Nstream_time : Long_Float;
        Bandwidth : Long_Float;

    begin

-- initialization

        for I in 1..Order Loop
            for J in 1..Order Loop
                A(I,J) := Float(0);
                B(I,J) := Float(2);
            end Loop;
        end Loop;
     
-- run the experiment

        for K in 0..Iterations Loop
     
            if K = 1 then
                T0 := Clock;
            end if;
     
            for I in 1..Order Loop
                A(I,J) := A(I,J) + B(I,J);
            end Loop;
     
        end Loop;
        T1 := Clock;
        DT := T1 - T0;

-- validation

        for I in 1..Order Loop
            for J in 1..Order Loop
                Asum := Asum + ABS ( A(I,J) );
            end Loop;
        end Loop;

        if Asum /= 0.0 then
            Put_Line("Asum=" & Float'Image(Asum) );
        else
            Put_Line("Solution validates");
            Bytes := Long_Integer(Order) * Long_Integer(Order) * Long_Float'Size / 4;
            Nstream_us := DT / US; -- this is per iteration now, thanks to US
            Nstream_time := Long_Float(Nstream_us) / Long_Float(1000000);
            Put_Line("Avg time (s): " & Long_Float'Image(Nstream_time));
            Bandwidth := Long_Float(Bytes) / Long_Float(Nstream_us);
            Put_Line("Rate (MB/s): " & Long_Float'Image(Bandwidth));
            -- archived for posterity
            --Put_Line("Bytes=" & Long_Integer'Image(Bytes) );
            --AvgTime := To_Duration(DT);
            --Put_Line("Total Time: " & Duration'Image(AvgTime) & " seconds");
            --Put_Line("Integer'Last=" & Integer'Image(Integer'Last) );
            --Put_Line("Long_Integer'Last=" & Long_Integer'Image(Long_Integer'Last) );
        end if;

    end;

end transpose;

