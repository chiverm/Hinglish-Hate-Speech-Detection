import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import os


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If file does not exist
        pd.errors.ParserError: If CSV cannot be parsed
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"✓ Loaded CSV: {filepath}")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        raise pd.errors.ParserError(f"Error parsing CSV file: {str(e)}")


def standardize_columns(df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    """
    Rename and standardize dataframe columns to 'text' and 'label'.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing labels
        
    Returns:
        pd.DataFrame: Dataframe with standardized columns
        
    Raises:
        ValueError: If specified columns don't exist
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in dataframe. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe. Available: {list(df.columns)}")
    
    df = df[[text_column, label_column]].copy()
    df.columns = ['text', 'label']
    print(f"✓ Standardized columns to 'text' and 'label'")
    
    return df


def handle_missing_values(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Handle missing values in the text column by removing rows with null/empty text.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of text column. Default: 'text'
        
    Returns:
        pd.DataFrame: Dataframe with missing values removed
    """
    initial_rows = len(df)
    
    # Remove rows where text is null
    df = df.dropna(subset=[text_column])
    
    # Remove rows where text is empty string or only whitespace
    df = df[df[text_column].str.strip().str.len() > 0]
    
    removed_rows = initial_rows - len(df)
    print(f"✓ Handled missing values: Removed {removed_rows} rows with empty/null text")
    print(f"  Remaining rows: {len(df)}")
    
    return df


def convert_labels_to_binary(df: pd.DataFrame, label_column: str = 'label', 
                            hate_labels: Optional[list] = None) -> pd.DataFrame:
    """
    Convert labels to binary format (0 = non-hate, 1 = hate).
    
    Handles common label variations:
    - 'hate', 'offensive', '1', 1 → 1 (hate speech)
    - 'normal', 'clean', 'non-hate', '0', 0 → 0 (not hate speech)
    
    Args:
        df (pd.DataFrame): Input dataframe
        label_column (str): Name of label column. Default: 'label'
        hate_labels (list, optional): Custom list of values to map to 1 (hate).
                                     If None, uses default common labels.
        
    Returns:
        pd.DataFrame: Dataframe with binary labels
        
    Raises:
        ValueError: If labels cannot be mapped to binary format
    """
    if hate_labels is None:
        # Common label values for hate speech
        hate_labels = ['hate', 'offensive', '1', 1, 'abusive', 'hateful']
    
    df = df.copy()
    
    # Convert labels to string for comparison (case-insensitive)
    label_mapping = {}
    for label in df[label_column].unique():
        if pd.isna(label):
            label_mapping[label] = np.nan
        elif str(label).lower() in [str(x).lower() for x in hate_labels]:
            label_mapping[label] = 1
        else:
            label_mapping[label] = 0
    
    df[label_column] = df[label_column].map(label_mapping)
    
    # Remove any rows with NaN labels after mapping
    df = df.dropna(subset=[label_column])
    
    unique_labels = sorted(df[label_column].unique())
    print(f"✓ Converted labels to binary format")
    print(f"  Unique labels: {unique_labels}")
    print(f"  Label distribution:\n{df[label_column].value_counts().to_string()}")
    
    return df


def split_train_test(df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label',
                     test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train and test sets (80/20 by default).
    
    Args:
        df (pd.DataFrame): Input dataframe with text and label columns
        text_column (str): Name of text column. Default: 'text'
        label_column (str): Name of label column. Default: 'label'
        test_size (float): Proportion of dataset to include in test split. Default: 0.2
        random_state (int): Random seed for reproducibility. Default: 42
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    X = df[text_column].values
    y = df[label_column].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Split dataset (test_size={test_size}):")
    print(f"  X_train shape: {X_train.shape} | y_train distribution:\n{pd.Series(y_train).value_counts().to_string()}")
    print(f"  X_test shape:  {X_test.shape} | y_test distribution:\n{pd.Series(y_test).value_counts().to_string()}")
    
    return X_train, X_test, y_train, y_test


def load_and_preprocess_dataset(
    filepath: str,
    text_column: str,
    label_column: str,
    hate_labels: Optional[list] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete pipeline to load, preprocess, and split a hate speech dataset.
    
    This function performs all preprocessing steps:
    1. Load CSV file
    2. Standardize column names
    3. Handle missing values
    4. Convert labels to binary
    5. Split into train/test sets
    
    Args:
        filepath (str): Path to the CSV file
        text_column (str): Name of the text column in the CSV
        label_column (str): Name of the label column in the CSV
        hate_labels (list, optional): Custom list of hate speech labels
        test_size (float): Test set proportion. Default: 0.2
        random_state (int): Random seed. Default: 42
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
        ...     'data/hate_speech.csv', 
        ...     'tweet', 
        ...     'label'
        ... )
    """
    print("\n" + "=" * 80)
    print("LOADING AND PREPROCESSING DATASET")
    print("=" * 80 + "\n")
    
    # Load CSV
    df = load_csv(filepath)
    print()
    
    # Standardize columns
    df = standardize_columns(df, text_column, label_column)
    print()
    
    # Handle missing values
    df = handle_missing_values(df)
    print()
    
    # Convert labels to binary
    df = convert_labels_to_binary(df, hate_labels=hate_labels)
    print()
    
    # Split train/test
    X_train, X_test, y_train, y_test = split_train_test(
        df, test_size=test_size, random_state=random_state
    )
    print()
    
    return X_train, X_test, y_train, y_test


def load_hinglish_dataset(
    save_csv: Optional[str] = None,
    combine_with: Optional[str] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load the findnitai/english-to-hinglish dataset from HuggingFace Hub.

    The dataset contains English ↔ Hinglish translation pairs. Since it has
    no hate-speech labels, the Hinglish sentences are treated as label=0
    (non-hate) and can optionally be combined with an existing labeled CSV
    to enrich the training data with Hinglish-language negative examples.

    Args:
        save_csv (str, optional): If provided, save the resulting DataFrame
            to this CSV path.
        combine_with (str, optional): Path to an existing labeled CSV
            (must have 'text' and 'label' columns) to concatenate with the
            Hinglish non-hate samples.
        sample_size (int, optional): If set, randomly sample this many rows
            from the Hinglish dataset before combining.
        random_state (int): Random seed for sampling. Default: 42.

    Returns:
        pd.DataFrame: DataFrame with columns ['text', 'label'].
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install it with:\n"
            "  pip install datasets"
        )

    print("⬇  Loading findnitai/english-to-hinglish from HuggingFace Hub...")
    ds = hf_load_dataset("findnitai/english-to-hinglish")
    train_split = ds["train"]

    hinglish_texts = [item["translation"]["hi_ng"] for item in train_split]
    df_hinglish = pd.DataFrame({"text": hinglish_texts, "label": 0})

    print(f"✓ Loaded {len(df_hinglish):,} Hinglish sentences (label=0 / non-hate)")

    if sample_size is not None and sample_size < len(df_hinglish):
        df_hinglish = df_hinglish.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        print(f"  Sampled down to {len(df_hinglish):,} rows")

    if combine_with is not None:
        df_existing = load_csv(combine_with)
        if "text" not in df_existing.columns or "label" not in df_existing.columns:
            raise ValueError(
                f"'{combine_with}' must have 'text' and 'label' columns. "
                f"Found: {list(df_existing.columns)}"
            )
        df_combined = pd.concat([df_existing, df_hinglish], ignore_index=True)
        df_combined = df_combined.dropna(subset=["text"])
        df_combined = df_combined[df_combined["text"].str.strip().str.len() > 0]
        print(f"✓ Combined dataset size: {len(df_combined):,} rows")
        label_counts = df_combined["label"].value_counts().to_dict()
        print(f"  Label distribution: {label_counts}")
        df_result = df_combined
    else:
        df_result = df_hinglish

    if save_csv is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv)), exist_ok=True)
        df_result.to_csv(save_csv, index=False)
        print(f"✓ Saved to {save_csv}")

    return df_result


if __name__ == "__main__":
    # Example usage with sample data
    print("\n" + "=" * 80)
    print("CREATING SAMPLE DATASET FOR DEMONSTRATION")
    print("=" * 80 + "\n")
    
    # Create sample data
    sample_data = {
        'tweet': [
            'I love this movie',
            'You are so stupid',
            'Great day today!',
            'I hate you all',
            'This is amazing',
            None,  # Missing value
            'Go die',
            'Beautiful sunset',
            '',  # Empty text
            'Seriously hate this',
        ],
        'sentiment': [
            'clean',
            'hate',
            'clean',
            'hate',
            'clean',
            'clean',
            'offensive',
            'clean',
            'clean',
            'hate',
        ]
    }
    
    # Save sample data to CSV
    sample_df = pd.DataFrame(sample_data)
    sample_csv_path = 'sample_hate_speech.csv'
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"Created sample dataset at: {sample_csv_path}\n")
    
    # Load and preprocess dataset
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
            filepath=sample_csv_path,
            text_column='tweet',
            label_column='sentiment',
            hate_labels=['hate', 'offensive', 'abusive'],
            test_size=0.2,
            random_state=42
        )
        
        print("\n" + "=" * 80)
        print("SAMPLE DATA FROM SPLITS")
        print("=" * 80 + "\n")
        
        print("Training samples (first 3):")
        for i in range(min(3, len(X_train))):
            print(f"  [{y_train[i]}] {X_train[i]}")
        
        print("\nTest samples (first 3):")
        for i in range(min(3, len(X_test))):
            print(f"  [{y_test[i]}] {X_test[i]}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up sample file
        if os.path.exists(sample_csv_path):
            os.remove(sample_csv_path)
            print(f"\n✓ Cleaned up temporary file: {sample_csv_path}")
