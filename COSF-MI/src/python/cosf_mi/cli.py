"""
COSF-MI: Command Line Interface

Provides CLI commands for COSF analysis.
"""

import sys
from typing import Optional
from pathlib import Path

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    TYPER_AVAILABLE = True
except ImportError:
    TYPER_AVAILABLE = False

from .core import print_constants, verify_cosf_convergence

if TYPER_AVAILABLE:
    app = typer.Typer(
        name="cosf-mi",
        help="COSF-MI: Mechanistic Interpretability Through Cosmological Synchronization",
    )
    console = Console()

    @app.command()
    def info():
        """Display COSF constants and verification."""
        print_constants()
        
        console.print("\n[bold green]Verification:[/bold green]")
        result = verify_cosf_convergence()
        
        table = Table(title="COSF Convergence Verification")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in result.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.6f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)

    @app.command()
    def analyze(
        model_name: str = typer.Argument(..., help="HuggingFace model name or path"),
        text: str = typer.Option(None, "--text", "-t", help="Text to analyze"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
        device: str = typer.Option("cuda", "--device", "-d", help="Device (cuda/cpu)"),
    ):
        """Analyze a model using COSF principles."""
        import torch
        
        console.print(f"[bold]Loading model:[/bold] {model_name}")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
                console.print("[green]Using CUDA[/green]")
            else:
                console.print("[yellow]Using CPU[/yellow]")
            
            from .analysis import CircuitExtractor
            
            extractor = CircuitExtractor(model, tokenizer, device=device)
            
            if text:
                console.print(f"\n[bold]Analyzing:[/bold] {text[:50]}...")
                result = extractor.analyze(text=text)
            else:
                console.print("[yellow]No text provided, using default[/yellow]")
                result = extractor.analyze(text="The quick brown fox jumps over the lazy dog.")
            
            # Display results
            extractor.print_summary(result)
            
            if output:
                import json
                with open(output, 'w') as f:
                    json.dump({
                        "global_metrics": result.global_metrics,
                        "layer_summaries": result.layer_summaries,
                        "n_circuits": len(result.circuits),
                    }, f, indent=2)
                console.print(f"\n[green]Results saved to {output}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    @app.command()
    def test():
        """Run basic tests to verify installation."""
        console.print("[bold]Running COSF-MI tests...[/bold]\n")
        
        # Test imports
        tests_passed = 0
        tests_failed = 0
        
        try:
            from .core import PHI, COSF, verify_cosf_convergence
            result = verify_cosf_convergence()
            assert result["is_valid"], "COSF convergence validation failed"
            console.print("[green]✓[/green] Core constants loaded")
            tests_passed += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Core constants: {e}")
            tests_failed += 1
        
        try:
            from .core import PhiDecomposer
            import torch
            decomposer = PhiDecomposer(use_gpu=False)
            test_tensor = torch.randn(4, 64)
            result = decomposer.analyze(test_tensor)
            assert result.phi_coherence >= 0
            console.print("[green]✓[/green] φ-decomposition working")
            tests_passed += 1
        except Exception as e:
            console.print(f"[red]✗[/red] φ-decomposition: {e}")
            tests_failed += 1
        
        try:
            from .core import HarmonicAnalyzer
            import torch
            analyzer = HarmonicAnalyzer()
            test_tensor = torch.randn(1, 128, 64)
            result = analyzer.analyze_sequence(test_tensor)
            console.print("[green]✓[/green] Harmonic analysis working")
            tests_passed += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Harmonic analysis: {e}")
            tests_failed += 1
        
        try:
            from .core import JuliaStabilityAnalyzer
            import torch
            analyzer = JuliaStabilityAnalyzer()
            test_tensor = torch.randn(4, 32, 64)
            result = analyzer.analyze_direct(test_tensor)
            assert 0 <= result.stability_score <= 1
            console.print("[green]✓[/green] Julia stability analysis working")
            tests_passed += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Julia stability: {e}")
            tests_failed += 1
        
        try:
            from .hooks import ModelHooks
            import torch.nn as nn
            model = nn.Linear(64, 64)
            hooks = ModelHooks(model)
            console.print("[green]✓[/green] Model hooks working")
            tests_passed += 1
        except Exception as e:
            console.print(f"[red]✗[/red] Model hooks: {e}")
            tests_failed += 1
        
        # Summary
        console.print(f"\n[bold]Results:[/bold] {tests_passed} passed, {tests_failed} failed")
        
        if tests_failed > 0:
            raise typer.Exit(1)

    def main():
        app()

else:
    # Fallback without typer
    def main():
        print("COSF-MI CLI")
        print("=" * 40)
        
        if len(sys.argv) > 1:
            cmd = sys.argv[1]
            if cmd == "info":
                print_constants()
                print("\nVerification:", verify_cosf_convergence())
            elif cmd == "test":
                print("Running basic tests...")
                from .core import PHI, COSF
                print(f"PHI = {PHI}")
                print(f"COSF = {COSF}")
                print("Basic imports working!")
            else:
                print(f"Unknown command: {cmd}")
                print("Available: info, test")
        else:
            print("Usage: python -m cosf_mi.cli <command>")
            print("Commands: info, test, analyze")
            print("\nFor full CLI, install: pip install typer rich")


if __name__ == "__main__":
    main()
