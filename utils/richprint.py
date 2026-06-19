from rich import print as rprint

plain_red = lambda x: f'[red]{x}[/red]'

bold_cyan = lambda x: f"[bold cyan]{x}[/bold cyan]"
bold_red = lambda x: f"[bold red]{x}[/bold red]"
bold_yellow = lambda x: f'[bold yellow]{x}[/bold yellow]'

printBoldCyan = lambda x: rprint(bold_cyan(x))
printBoldYellow = lambda x: rprint(bold_yellow(x))