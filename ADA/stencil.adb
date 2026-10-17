with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Real_Time,
    Ada.Command_line;

procedure stencil is

    use
        Ada.Text_IO,
        Ada.Integer_Text_IO,
        Ada.Real_Time,
        Ada.Command_line;

    -- GNAT Integer = int32_t, Long_Integer = int64_t
    Iterations : Natural := 10;
    N : Natural := 1_000;
    Radius : Natural := 2;
    Is_Star : Boolean := True;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial stencil execution on 2D grid");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        N := Integer'Value(Argument(2));
    end if;
    if Argument_Count > 2 then
        Is_Star := Argument(3) = "star";
    end if;
    if Argument_Count > 3 then
        Radius := Integer'Value(Argument(4));
    end if;

    if Iterations < 1 then
        Put_Line("Iteration count must be greater than " & Integer'Image(Iterations) );
    end if;
    if 2*Radius + 1 > N then
        Put_Line("ERROR: Stencil radius exceeds grid size");
        return;
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Grid size             =" & Integer'Image(N) );
    if Is_Star then
        Put_Line("Type of stencil       = star");
    else
        Put_Line("Type of stencil       = grid");
    end if;
    Put_Line("Radius of stencil     =" & Integer'Image(Radius) );

    declare

        -- Double precision throughout: Ada.Numerics.Real_Arrays.Real_Matrix
        -- is single-precision Float, which does not have enough mantissa
        -- bits to hold the L1-norm accumulation below to 1.0E-8 once the
        -- stencil has more than a handful of points (grid pattern, or a
        -- larger radius).
        -- Weight is small (a (2*Radius+1)^2 stencil) and stays on the
        -- stack; Input/Output are heap-allocated (like nstream_array.adb)
        -- since an N-by-N Long_Float grid at realistic benchmark sizes
        -- (e.g. 1000x1000 = 8MB) does not reliably fit on the default
        -- 8MB stack.
        type Real_Matrix is array(Integer range <>, Integer range <>) of Long_Float;
        type Real_Matrix_Access is access Real_Matrix;
        Weight : Real_Matrix(-Radius..Radius,-Radius..Radius) := (others => (others => 0.0));
        Input : Real_Matrix_Access := new Real_Matrix(0..N-1,0..N-1);
        Output : Real_Matrix_Access := new Real_Matrix(0..N-1,0..N-1);

        T0, T1 : Time;
        DT : Time_Span;

        US : constant Time_Span := Microseconds(US => Iterations);

        K : Integer := 0;

        StencilSize : Integer;
        Norm : Long_Float := 0.0;
        ActivePoints : constant Long_Float := Long_Float((N - 2*Radius) * (N - 2*Radius));
        ReferenceNorm : constant Long_Float := Long_Float(2 * (Iterations+1));
        Epsilon : constant Long_Float := 1.0E-8;

        AvgTime : Long_Float;
        Nstream_us : Integer;

    begin

-- initialize the input array: in(i,j) = i+j, same convention as the other
-- language ports of this kernel

        for I in 0..N-1 Loop
            for J in 0..N-1 Loop
                Input(I,J) := Long_Float(I+J);
                Output(I,J) := 0.0;
            end Loop;
        end Loop;

-- fill the weight stencil: a discrete approximation to the derivative
-- operator, same construction as the other language ports

        if Is_Star then
            StencilSize := 4*Radius + 1;
            for I in 1..Radius Loop
                Weight(0,I)  := 1.0 / Long_Float(2*I*Radius);
                Weight(I,0)  := 1.0 / Long_Float(2*I*Radius);
                Weight(0,-I) := -1.0 / Long_Float(2*I*Radius);
                Weight(-I,0) := -1.0 / Long_Float(2*I*Radius);
            end Loop;
        else
            StencilSize := (2*Radius+1) * (2*Radius+1);
            for J in 1..Radius Loop
                for I in -J+1..J-1 Loop
                    Weight(I,J)  := 1.0 / Long_Float(4*J*(2*J-1)*Radius);
                    Weight(I,-J) := -1.0 / Long_Float(4*J*(2*J-1)*Radius);
                    Weight(J,I)  := 1.0 / Long_Float(4*J*(2*J-1)*Radius);
                    Weight(-J,I) := -1.0 / Long_Float(4*J*(2*J-1)*Radius);
                end Loop;
                Weight(J,J)   := 1.0 / Long_Float(4*J*Radius);
                Weight(-J,-J) := -1.0 / Long_Float(4*J*Radius);
            end Loop;
        end if;

-- run the experiment

        for K in 0..Iterations Loop

            if K = 1 then
                T0 := Clock;
            end if;

            -- apply the stencil operator
            for I in Radius..N-1-Radius Loop
                for J in Radius..N-1-Radius Loop
                    if Is_Star then
                        for JJ in -Radius..Radius Loop
                            Output(I,J) := Output(I,J) + Weight(0,JJ) * Input(I,J+JJ);
                        end Loop;
                        for II in -Radius..-1 Loop
                            Output(I,J) := Output(I,J) + Weight(II,0) * Input(I+II,J);
                        end Loop;
                        for II in 1..Radius Loop
                            Output(I,J) := Output(I,J) + Weight(II,0) * Input(I+II,J);
                        end Loop;
                    else
                        for JJ in -Radius..Radius Loop
                            for II in -Radius..Radius Loop
                                Output(I,J) := Output(I,J) + Weight(II,JJ) * Input(I+II,J+JJ);
                            end Loop;
                        end Loop;
                    end if;
                end Loop;
            end Loop;

            -- add a constant to the input array to force a refresh of the
            -- neighbor data on the next iteration
            for I in 0..N-1 Loop
                for J in 0..N-1 Loop
                    Input(I,J) := Input(I,J) + 1.0;
                end Loop;
            end Loop;

        end Loop;
        T1 := Clock;
        DT := T1 - T0;

-- validation: L1 norm of the interior of the output array should match a
-- known closed-form reference value

        for I in Radius..N-1-Radius Loop
            for J in Radius..N-1-Radius Loop
                Norm := Norm + abs ( Output(I,J) );
            end Loop;
        end Loop;
        Norm := Norm / ActivePoints;

        if abs ( Norm - ReferenceNorm ) > Epsilon then
            Put_Line("ERROR: L1 norm = " & Long_Float'Image(Norm) & " Reference L1 norm = " & Long_Float'Image(ReferenceNorm) );
        else
            Put_Line("Solution validates");
            Nstream_us := DT / US; -- average time per iteration, in microseconds, thanks to US
            AvgTime := Long_Float(Nstream_us) / Long_Float(1_000_000);
            Put_Line("Avg time (s): " & Long_Float'Image(AvgTime) );
            Put_Line("Rate (MFlops/s): " & Long_Float'Image(1.0E-6 * Long_Float(2*StencilSize+1) * ActivePoints / AvgTime) );
        end if;

    end;

end stencil;
