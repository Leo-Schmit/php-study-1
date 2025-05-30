<?php
require './util.php';

$code = <<<'CODE'
<?php
$name = 'world';
echo 'Hello ' . $name . '!';
CODE;

$ast = ast\parse_code($code, $version = 100);

echo ast_dump($ast), "\n";

$first_stmt = $ast->children[0] ?? null;
if (
    $first_stmt instanceof ast\Node &&
    $first_stmt->kind === ast\AST_ASSIGN &&
    isset($first_stmt->children['expr']) &&
    (
        is_string($first_stmt->children['expr']) ||
        ($first_stmt->children['expr'] instanceof ast\Node && $first_stmt->children['expr']->kind === ast\AST_CONST)
    )
) {
    echo "\n\$name is assigned a string.";
} else {
    echo "\n\$name is not assigned a string.";
}
