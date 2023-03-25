with
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Command_line;

use
    Ada.Text_IO,
    Ada.Integer_Text_IO,
    Ada.Command_line;

procedure Args is

    A : Integer := 1;
    I : Integer := 10;
    N : Integer := 1_000_000;

begin

    Put_Line("Args, World!");

    Put("Argument_Count=");
    Put(Item => Argument_Count, Width => 1);
    Put_Line("");

    if Argument_Count > 0 then
        Put_Line(Item => Argument(1));
    end if;
    if Argument_Count > 1 then
        Put_Line(Item => Argument(2));
    end if;

end Args;

