class DataManager:
    # def __init__(self, df):
    #     self.df = df

    def normalize_data(self, df, input_columns, output_column):
        max_list = []
        for field in input_columns:
            max_list.append(max(df[field]))
            df[field] = df[field] / max(df[field])

        cardio_disease_info = df[input_columns].to_numpy()
        has_cardio = df[output_column].to_numpy()
        return cardio_disease_info, has_cardio, max_list

    def convert_user_data(self, gender, cholesterol, glucose, smoke, alcohol, active):
        if gender == "Female":
            gender = 1
        else:
            gender = 2

        if cholesterol == "less than 200 mg/dL":
            cholesterol = 1
        elif cholesterol == "200 to 239 mg/dL":
            cholesterol = 2
        else:
            cholesterol = 3

        if glucose == "less than 100 mg/dL":
            glucose = 1
        elif glucose == "100 to 125 mg/dL":
            glucose = 2
        else:
            glucose = 3

        if smoke == "No":
            smoke = 0
        else:
            smoke = 1

        if alcohol == "No":
            alcohol = 0
        else:
            alcohol = 1

        if active == "No":
            active = 0
        else:
            active = 1

        return gender, cholesterol, glucose, smoke, alcohol, active

