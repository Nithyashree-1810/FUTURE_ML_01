import pandas as pd
from prophet import Prophet

INPUT_FILE_OVERALL = r"C:\Users\fidel\Downloads\forecast_overall.csv"
INPUT_FILE_CAT = r"C:\Users\fidel\Downloads\forecast_by_category.csv"
INPUT_FILE_SUBCAT = r"C:\Users\fidel\Downloads\forecast_by_subcategory.csv"

OUTPUT_COMBINED = r"C:\Users\fidel\Downloads\forecast_combined.csv"
FORECAST_MONTHS = 12


def run_forecast(df, date_col, value_col, level, group_col=None):
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").fillna(0)

    results = []

    if group_col:
        groups = df[group_col].unique()
        for g in groups:
            sub = df[df[group_col] == g]
            df_monthly = (
                sub.groupby(pd.Grouper(key=date_col, freq="M"))
                   .agg({value_col: "sum"})
                   .reset_index()
            )
            df_prophet = df_monthly.rename(columns={date_col: "ds", value_col: "y"})

            if len(df_prophet) < 12:  
                continue

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq="M")
            forecast = model.predict(future)

        
            forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]]
            forecast_out = forecast_out.merge(df_prophet, on="ds", how="left")
            forecast_out["level"] = level
            forecast_out["group_name"] = g
            results.append(forecast_out)

        return pd.concat(results, ignore_index=True)

    else:  
        df_monthly = (
            df.groupby(pd.Grouper(key=date_col, freq="M"))
               .agg({value_col: "sum"})
               .reset_index()
        )
        df_prophet = df_monthly.rename(columns={date_col: "ds", value_col: "y"})

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=FORECAST_MONTHS, freq="M")
        forecast = model.predict(future)

        forecast_out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "weekly", "yearly"]]
        forecast_out = forecast_out.merge(df_prophet, on="ds", how="left")
        forecast_out["level"] = level
        forecast_out["group_name"] = "all"

        return forecast_out


df_overall = pd.read_csv(INPUT_FILE_OVERALL)
df_cat = pd.read_csv(INPUT_FILE_CAT)
df_subcat = pd.read_csv(INPUT_FILE_SUBCAT)


overall_forecast = run_forecast(df_overall, "date", "forecast_sales", "overall")
cat_forecast = run_forecast(df_cat, "date", "forecast_sales", "category", group_col="category")
subcat_forecast = run_forecast(df_subcat, "date", "forecast_sales", "subcategory", group_col="sub-category")


forecast_combined = pd.concat([overall_forecast, cat_forecast, subcat_forecast], ignore_index=True)


forecast_combined.rename(columns={
    "ds": "date",
    "yhat": "forecast_sales",
    "yhat_lower": "forecast_lower",
    "yhat_upper": "forecast_upper",
    "y": "actual_sales"
}, inplace=True)


forecast_combined.to_csv(OUTPUT_COMBINED, index=False)
print(f" Combined forecast with trend & seasonality saved to {OUTPUT_COMBINED}")
