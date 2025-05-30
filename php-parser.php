<?php

require 'vendor/autoload.php';

use PhpParser\ParserFactory;
use PhpParser\NodeDumper;

$code = <<<'CODE'
<?php
$name = 'world';
echo 'Hello ' . $name . '!';
CODE;
$parser = (new ParserFactory())->createForNewestSupportedVersion();
$ast = $parser->parse($code);
$dumper = new NodeDumper;
echo $dumper->dump($ast);

$assign = $ast[0]->expr;
var_dump($assign);
if ($assign instanceof PhpParser\Node\Expr\Assign && $assign->expr instanceof PhpParser\Node\Scalar\String_) {
    echo "\n\$name is assigned a string.";
} else {
    echo "\n\$name is not assigned a string.";
}
