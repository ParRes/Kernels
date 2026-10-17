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

    -- GNAT Integer = int32_t, Long_Integer = int64_t
    Iterations : Integer := 10;
    Length : Long_Integer := 1_000_000;

    Scalar : constant := 3.0;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial STREAM triad: A = B + scalar * C");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        Length := Long_Integer'Value(Argument(2));
    end if;

    if Iterations < 1 then
        Put_Line("Iteration count must be greater than " & Integer'Image(Iterations) );
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Vector length        =" & Long_Integer'Image(Length) );

    declare
        -- Heap-allocated (via an access type), not a plain local array:
        -- a stack-declared array of this size does not reliably fit on
        -- the default 8MB stack once Length grows past a few hundred
        -- thousand elements. Double precision throughout, matching the
        -- "double precision" convention of the other language ports.
        type Long_Float_Array is array(Long_Integer range <>) of Long_Float
            with Default_Component_Value => 0.0;
        type Long_Float_Array_Access is access Long_Float_Array;
        A : Long_Float_Array_Access := new Long_Float_Array(1..Length);
        B : Long_Float_Array_Access := new Long_Float_Array(1..Length);
        C : Long_Float_Array_Access := new Long_Float_Array(1..Length);

        T0, T1 : Time;
        DT : Time_Span;

        US : constant Time_Span := Microseconds(US => Iterations);

        K : Integer := 0;
        AR : Long_Float := 0.0;
        BR : Long_Float := 2.0;
        CR : Long_Float := 2.0;
        Asum : Long_Float := 0.0;

        Bytes : Long_Integer;
        Nstream_us : Integer;
        Nstream_time : Long_Float;
        Bandwidth : Long_Float;

    begin

-- initialization

        for I in 1..Length Loop
            A(I) := 0.0;
            B(I) := 2.0;
            C(I) := 2.0;
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
            Put_Line("Asum=" & Long_Float'Image(Asum) );
        else
            Put_Line("Solution validates");
            -- 4 array touches (read A,B,C + write A) * 8 bytes each
            Bytes := Length * Long_Float'Size / 2;
            Nstream_us := DT / US; -- this is per iteration now, thanks to US
            Nstream_time := Long_Float(Nstream_us) / Long_Float(1_000_000);
            Put_Line("Avg time (s): " & Long_Float'Image(Nstream_time));
            Bandwidth := Long_Float(Bytes) / Long_Float(Nstream_us);
            Put_Line("Rate (MB/s): " & Long_Float'Image(Bandwidth));
        end if;

    end;

end Nstream;
