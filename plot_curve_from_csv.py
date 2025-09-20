def plot_curve_from_csv(csv_path, row=0, image_name=None, title=None):
    """
    Plot a single curve from a CSV.
    - Training labels format: columns y1_norm..y20_norm (normalized to [0,1])
    - Inference format: columns (x1,y1)..(x20,y20)
    Select by row index or image filename (if there's an 'image' column).
    """
    df = pd.read_csv(csv_path)
    if image_name is not None and 'image' in df.columns:
        sel = df[df['image'] == image_name]
        if sel.empty:
            raise ValueError(f"image '{image_name}' not found in CSV")
        row_data = sel.iloc[0]
    else:
        row_data = df.iloc[row]

    # Try training-labels format first
    y_norm_cols = [c for c in df.columns if re.fullmatch(r'y\d+_norm', c)]
    if y_norm_cols:
        y_norm_cols = sorted(y_norm_cols, key=lambda c: int(re.findall(r'\d+', c)[0]))
        y_norm = row_data[y_norm_cols].to_numpy(dtype=float)
        y = 1 + 19*y_norm
        x = np.arange(1, len(y)+1, dtype=float)
    else:
        # Fallback: inference format "(xk,yk)" columns
        pair_cols = [c for c in df.columns if re.fullmatch(r'\(x\d+,y\d+\)', c)]
        if not pair_cols:
            raise ValueError("CSV format not recognized: expected y*_norm or (xk,yk) columns.")
        pair_cols = sorted(pair_cols, key=lambda c: int(re.findall(r'\d+', c)[0]))
        xs, ys = [], []
        for c in pair_cols:
            val = str(row_data[c])
            m = re.match(r'\(?\s*([-\d\.eE]+)\s*,\s*([-\d\.eE]+)\s*\)?', val)
            if not m:
                raise ValueError(f"Bad pair value in column {c}: {val}")
            xs.append(float(m.group(1))); ys.append(float(m.group(2)))
        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)

    plt.figure(figsize=(4,4))
    plt.plot(x, y, marker='o')
    plt.xlim(1, 20); plt.ylim(1, 20)
    plt.xlabel('x (axis units)'); plt.ylabel('y (axis units)')
    plt.title(title or image_name or f'Row {row}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()
