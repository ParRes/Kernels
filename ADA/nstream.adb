with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    --Ada.Strings,
    --Ada.Strings.Bounded,
    Ada.Real_Time,
    Ada.Command_line;

use
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    --Ada.Strings,
    --Ada.Strings.Bounded,
    Ada.Real_Time,
    Ada.Command_line;

procedure Nstream is

    Iterations : Integer := 10;
    Length : Integer := 1_000_000;

    Scalar : constant := 3.0;

    type Vector is array(Integer range <>) of Float;

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
        I : Integer := 0;
        A : Vector(1..Length);
        B : Vector(1..Length);
        C : Vector(1..Length);

        T0, T1 : Time;
        Nstream_Time : Time_Span;

        K : Integer := 0;
        AR : Float := 0.0;
        BR : Float := 2.0;
        CR : Float := 2.0;
        Asum : Float := 0.0;

        AvgTime : Duration;
        Bytes : Integer := Float'Size / 8;
        --Temp : Integer := 1;

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
        Nstream_Time := T1 - T0;

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
            Put_Line(Integer'Image(Nstream_Time / Time_Span_Unit) & " Time_Span_Units");
            --Temp := 1 / Time_Span_Unit;
            --Put_Line(Time_Span'Image(Time_Span_Unit) & " Time_Span_Unit");
            AvgTime := To_Duration(Nstream_Time);
            Put_Line("Bytes=" & Integer'Image(Bytes) );
            Put_Line("Time=" & Duration'Image(AvgTime) & " seconds");
            --Put(Bytes 
        end if;

    end;

end Nstream;

