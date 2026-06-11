$nb = Get-Content -Raw "Context_Aware_RAG_Chatbot.ipynb" | ConvertFrom-Json
$output = @()
foreach ($cell in $nb.cells) {
    if ($cell.cell_type -eq "markdown") {
        $output += $cell.source -join ""
    }
}
$output | Out-File -FilePath "markdown.txt" -Encoding utf8
