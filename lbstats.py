import pandas as pd
from pathlib import Path
from src.labels import LabelsEnum
import click

def analyze_labels(parquet_path: Path):
    if not parquet_path.exists():
        print(f"Error: File not found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)

    total_rows = len(df)
    print(f"Total samples: {total_rows}")
    print("-" * 50)
    print(f"{'Label Name':<25} | {'Count':<10} | {'Ratio':<10}")
    print("-" * 50)


    for label in LabelsEnum:
        # Check if the bit at position label.value is set
        # We use bitwise AND with a mask (1 shifted left by the label value)
        mask = 1 << label.value
        
        # Count how many rows have this bit set
        count = (df['MLBEncoded'] & mask).astype(bool).sum()
        ratio = count / total_rows if total_rows > 0 else None
                
        if ratio is not None:
            print(f"{label.name:<25} | {count:<10} | {ratio:.4f}")
        else:
            print(f"{label.name:<25} | {count:<10} | {'N/A':<10}")


@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
def main(input_path):
    input_path = Path(input_path).resolve()
    analyze_labels(input_path)


if __name__ == "__main__":
    main()
