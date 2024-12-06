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

procedure Args is

    package BS is new Ada.Strings.Bounded.Generic_Bounded_Length (Max => 15);
    use BS;

    S : Bounded_String;

    A : Integer := 1;
    I : Integer := 10;
    N : Integer := 1_000_000;

begin

    Put_Line("Args, World!");

    --Put("Argument_Count=");
    --Put(Item => Argument_Count, Width => 1);
    Put_Line("Argument_Count=" & Argument_Count'Image);

    if Argument_Count > 0 then
        Put_Line("Arg1=" & Argument(1));
    end if;
    if Argument_Count > 1 then
        Put_Line("Arg2=" & Argument(2));
    end if;

end Args;

