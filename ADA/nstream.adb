with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Real_Time,
    Ada.Command_line;

procedure Nstream is

    use
        Ada.Text_IO,
        Ada.Integer_Text_IO,
        Ada.Real_Time,
        Ada.Command_line;

    Iterations : Integer := 10;
    Length : Integer := 1_000_000;

    Scalar : constant := 3.0;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial STREAM triad: A = B + scalar * C");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        Length := Integer'Value(Argument(2));
    end if;

    if Iterations < 2 then
        Put_Line("Iteration count must be greater than " & Integer'Image(Iterations) );
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Vector length        =" & Integer'Image(Length) );

    declare
        type Vector is array(1..Length) of Float;

        I : Integer := 0;
        A : Vector;
        B : Vector;
        C : Vector;

        T0, T1 : Time;
        DT : Time_Span;

        US : constant Time_Span := Microseconds(US => Iterations);

        K : Integer := 0;
        AR : Float := 0.0;
        BR : Float := 2.0;
        CR : Float := 2.0;
        Asum : Float := 0.0;

        AvgTime : Duration;
        Bytes : Integer := Float'Size / 8;
        Nstream_us : Integer;
        Nstream_time : Float;
        Bandwidth : Float;

    begin

-- initialization

        for I in 1..Length Loop
            A(I) := Float(0);
            B(I) := Float(2);
            C(I) := Float(2);
        end Loop;
     
-- run the experiment

        for K in 0..Iterations Loop
     
            if K = 1 then
                T0 := Clock;
            end if;
     
            for I in 1..Length Loop
                A(I) := A(I) + B(I) + Scalar * C(I);
            end Loop;
     
        end Loop;
        T1 := Clock;
        DT := T1 - T0;

-- validation

        for K in 0..Iterations Loop
            AR := AR + BR + Scalar * CR;
        end Loop;

        for I in 1..Length Loop
            Asum := Asum + ABS ( A(I) - AR );
        end Loop;

        if Asum /= 0.0 then
            Put_Line("Asum=" & Float'Image(Asum) );
        else
            Put_Line("Solution validates");
            Bytes := Bytes * Length * 4;
            Nstream_us := DT / US; -- this is per iteration now, thanks to US
            Nstream_time := Float(Nstream_us) / Float(1000000);
            Put_Line("Avg time (s): " & Float'Image(Nstream_time));
            Bandwidth := Float(Bytes) / Float(Nstream_us);
            Put_Line("Rate (MB/s): " & Float'Image(Bandwidth));
            -- archived for posterity
            --Put_Line("Bytes=" & Integer'Image(Bytes) );
            AvgTime := To_Duration(DT);
            Put_Line("Total Time: " & Duration'Image(AvgTime) & " seconds");
        end if;

    end;

end Nstream;

