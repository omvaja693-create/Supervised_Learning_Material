# Streamlit Deployment Checklist

## Before deployment

- [ ] `python train_model.py` runs successfully
- [ ] `model/house_price_pipeline.pkl` exists
- [ ] `app.py` runs with `streamlit run app.py`
- [ ] `requirements.txt` exists
- [ ] Test prediction works

## GitHub repository

Upload:

```text
app.py
requirements.txt
model/house_price_pipeline.pkl
data/house_data.csv
README.md
```

`train_model.py` and the notebook are useful for teaching/retraining.

## Streamlit Community Cloud

1. Push the project to GitHub.
2. Create a new Streamlit app.
3. Select the repository and branch.
4. Set the main file to `app.py`.
5. Deploy.
6. Test the prediction form.

## Security

Do not put API keys, passwords, tokens, or other secrets into the repository.
