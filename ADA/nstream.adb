with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Strings,
    Ada.Strings.Bounded,
    Ada.Command_line;

use
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Strings,
    Ada.Strings.Bounded,
    Ada.Command_line;

procedure Nstream is

    Iterations : Integer := 10;
    Length : Integer := 1_000_000;

begin

    Put_Line("Parallel Research Kernels");
    Put_Line("Ada Serial STREAM triad: A = B + scalar * C");

    if Argument_Count > 0 then
        Iterations := Integer'Value(Argument(1));
    end if;
    if Argument_Count > 1 then
        Length := Integer'Value(Argument(2));
    end if;

    Put_Line("Number of iterations =" & Integer'Image(Iterations) );
    Put_Line("Vector length        =" & Integer'Image(Length) );

end Nstream;

