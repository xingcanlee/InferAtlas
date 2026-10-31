{
  function node(value, start = location().start) {
    options.positions?.set(value, start);
    return value;
  }
  function fold(left, tail) {
    return tail.reduce((left, [op, right]) =>
      node({ type: "binary", op, left, right }, options.positions?.get(left)), left);
  }
}
Start = _ expr:Conditional _ { return expr; }
Conditional = test:Or tail:(_ "?" _ yes:Conditional _ ":" _ no:Conditional { return [yes, no]; })? {
  return tail ? node({ type: "conditional", test, consequent: tail[0], alternate: tail[1] }) : test;
}
Or = left:And tail:(_ op:"||" _ right:And { return [op, right]; })* { return fold(left, tail); }
And = left:Equality tail:(_ op:"&&" _ right:Equality { return [op, right]; })* { return fold(left, tail); }
Equality = left:Comparison tail:(_ op:("==" / "!=") _ right:Comparison { return [op, right]; })* { return fold(left, tail); }
Comparison = left:Sum tail:(_ op:("<=" / ">=" / "<" / ">") _ right:Sum { return [op, right]; })* { return fold(left, tail); }
Sum = left:Product tail:(_ op:("+" / "-") _ right:Product { return [op, right]; })* { return fold(left, tail); }
Product = left:Unary tail:(_ op:("*" !"*" { return "*"; } / "/" / "%") _ right:Unary { return [op, right]; })* { return fold(left, tail); }
Unary = op:("+" / "-" / "!") _ argument:Unary { return node({ type: "unary", op, argument }); } / Power
Power = left:Primary tail:(_ "**" _ right:Unary { return right; })? {
  return tail ? node({ type: "binary", op: "**", left, right: tail }) : left;
}
Primary = Number / Boolean / Call / Identifier / "(" _ expr:Conditional _ ")" { return expr; }
Number = value:$(([0-9]+ ("." [0-9]*)? / "." [0-9]+) ([eE] [+-]? [0-9]+)?) {
  return node({ type: "number", value: Number(value) });
}
Boolean = value:$("true" / "false") ![a-zA-Z0-9_] { return node({ type: "boolean", value: value === "true" }); }
Call = name:$("max" / "min" / "ceil" / "floor" / "cdiv" / "round_up" / "bytes") ![a-zA-Z0-9_] _ "(" _ args:Arguments? _ ")" {
  return node({ type: "call", name, args: args ?? [] });
}
Arguments = first:Conditional rest:(_ "," _ expr:Conditional { return expr; })* { return [first, ...rest]; }
Identifier = name:$([a-zA-Z_] [a-zA-Z0-9_]*) { return node({ type: "identifier", name }); }
_ = [ \t\r\n]*
