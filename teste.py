# -*- coding: utf-8 -*-
"""
Dental Insurance Fraud Detection System
"""

import pandas as pd
import numpy as np
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



class DentalInsuranceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de decetção de fraudes")
        self.root.geometry("600x500")
        self.root.configure(bg="#8ec1e6")

        # Try to load the model if it exists, otherwise create a placeholder
        self.model = self.load_model()

        # Create a main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Detector de fraudes", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Input fields
        self.create_input_fields(main_frame)

        # Submit button
        submit_button = ttk.Button(main_frame, text="Analizar", command=self.predict)
        submit_button.grid(row=8, column=0, columnspan=2, pady=20)

        # Results frame
        result_frame = ttk.LabelFrame(main_frame, text="Resultado da predição")
        result_frame.grid(row=9, column=0, columnspan=2, pady=10, sticky="ew")

        self.result_label = ttk.Label(result_frame, text="Fill the form and click 'Analyze'", font=("Arial", 12))
        self.result_label.pack(pady=10)

        # Add a button to train the model
        train_button = ttk.Button(main_frame, text="Modelo treinado", command=self.train_model)
        train_button.grid(row=10, column=0, columnspan=2, pady=10)

    def create_input_fields(self, parent):
        # Age
        ttk.Label(parent, text="Idade:").grid(row=1, column=0, sticky="w", pady=5)
        self.age_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.age_var, width=30).grid(row=1, column=1, sticky="ew", pady=5)

        # Sex
        ttk.Label(parent, text="Sexo:").grid(row=2, column=0, sticky="w", pady=5)
        self.sex_var = tk.StringVar()
        sex_combo = ttk.Combobox(parent, textvariable=self.sex_var, width=28)
        sex_combo['values'] = ('masculino', 'feminino')
        sex_combo.grid(row=2, column=1, sticky="ew", pady=5)

        # BMI
        ttk.Label(parent, text="IMC:").grid(row=3, column=0, sticky="w", pady=5)
        self.bmi_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.bmi_var, width=30).grid(row=3, column=1, sticky="ew", pady=5)

        # Children
        ttk.Label(parent, text="Numero de filhos :").grid(row=4, column=0, sticky="w", pady=5)
        self.children_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.children_var, width=30).grid(row=4, column=1, sticky="ew", pady=5)

        # Smoker
        ttk.Label(parent, text="Fumante:").grid(row=5, column=0, sticky="w", pady=5)
        self.smoker_var = tk.StringVar()
        smoker_combo = ttk.Combobox(parent, textvariable=self.smoker_var, width=28)
        smoker_combo['values'] = ('Sim', 'Não')
        smoker_combo.grid(row=5, column=1, sticky="ew", pady=5)

        # Region
        ttk.Label(parent, text="Região:").grid(row=6, column=0, sticky="w", pady=5)
        self.region_var = tk.StringVar()
        region_combo = ttk.Combobox(parent, textvariable=self.region_var, width=28)
        region_combo['values'] = ('Nordeste', 'Norte', 'Sudeste', 'Sul')
        region_combo.grid(row=6, column=1, sticky="ew", pady=5)

        # Procedure Type (added)
        ttk.Label(parent, text="Tipo de Procedimento:").grid(row=7, column=0, sticky="w", pady=5)
        self.procedure_var = tk.StringVar()
        procedure_combo = ttk.Combobox(parent, textvariable=self.procedure_var, width=28)
        procedure_combo['values'] = ('Limpeza Dental', 'Aparelho Ortodôntico', 'Extração dental',
                                     'Canal', 'Aplicação de protese', 'Panoramica', 'Cirurgia')
        procedure_combo.grid(row=7, column=1, sticky="ew", pady=5)

    def load_model(self):
        # Check if the model file exists
        if os.path.exists('dental_fraud_model.pkl'):
            try:
                with open('dental_fraud_model.pkl', 'rb') as file:
                    return pickle.load(file)
            except:
                print("Error loading model file, creating a new one.")

        # If the model doesn't exist, create a placeholder model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model

    def prepare_input(self):
        try:
            age = float(self.age_var.get())
            sex = 1 if self.sex_var.get().lower() == 'male' else 0
            bmi = float(self.bmi_var.get())
            children = int(self.children_var.get())
            smoker = 1 if self.smoker_var.get().lower() == 'yes' else 0

            # Convert region to numerical value
            region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
            region = region_mapping.get(self.region_var.get().lower(), 0)

            # Convert procedure type to numerical value
            procedure_mapping = {
                'Limpeza Dental': 0,
                'Aparelho Ortodôntico': 1,
                'Extração dental': 2,
                'Canal': 3,
                'Aplicação de protese': 4,
                'Panoramica': 5,
                'Cirurgia': 6
            }
            procedure = procedure_mapping.get(self.procedure_var.get(), 0)

            # Calculate standard value for the procedure
            procedure_values = {
                'Limpeza Dental': 100,
                'Aparelho Ortodôntico': 1300,
                'Extração dental': 400,
                'Canal': 800,
                'Aplicação de protese': 3000,
                'Panoramica': 200,
                'Cirurgia': 5000
            }
            value = procedure_values.get(self.procedure_var.get(), 200)

            # Validation
            if age < 1 or age > 120:
                raise ValueError("A idade deve estár entre 1 e 120 anos")

            if bmi < 10 or bmi > 70:
                raise ValueError("O imc deve estar entre 10 e 70")

            if children < 0:
                raise ValueError("O numero de Filhos não pode ser negativo")

            # Return data as a pandas DataFrame with all required columns
            data = pd.DataFrame({
                'idade': [age],
                'sexo': [sex],
                'IMC': [bmi],
                'Filhos': [children],
                'Fumante': [smoker],
                'região': [region],
                'tipo_procedimento': [procedure],
                'valor_medico': [value],
                'valor_apolice': [value * (1 + np.random.uniform(-0.1, 0.1))],
                'frequencia': [np.random.randint(1, 5)]  # Random frequency between 1-4
            })
            return data

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return None

    def predict(self):
        # Get and prepare input data
        input_data = self.prepare_input()

        if input_data is not None:
            try:
                # If the model is trained, use it to predict
                prediction = self.model.predict(input_data)
                probability = self.model.predict_proba(input_data)

                # Display the prediction
                if prediction[0] == 1:
                    result_text = f"ALERT: Potential fraud detected!\nProbability: {probability[0][1]:.2f}"
                    self.result_label.config(text=result_text, foreground="red")
                else:
                    result_text = f"No fraud detected\nProbability of no fraud: {probability[0][0]:.2f}"
                    self.result_label.config(text=result_text, foreground="green")

            except Exception as e:
                # Model is not trained or there's a compatibility issue
                messagebox.showinfo("Infomações do modelo",
                                    "Este é uma aplicação destinada a predição, na aplicação real você deve:\n"
                                    "1. treinar seu modelo\n"
                                    "2. Salvar usando  pickle.dump(model, open('dental_fraud_model.pkl', 'wb'))\n"
                                    "3. O app  carregará automaticamente o modelo salvo")
                # Generate a random prediction for demonstration
                is_fraud = np.random.choice([True, False], p=[0.2, 0.8])
                fraud_prob = np.random.uniform(0.7, 0.95) if is_fraud else np.random.uniform(0.05, 0.3)

                if is_fraud:
                    result_text = f"ALERTA: Possivel fraudador\nProbability: {fraud_prob:.2f}\n(Demo Mode)"
                    self.result_label.config(text=result_text, foreground="red")
                else:
                    result_text = f"Sem fraude detectada\nProbability of no fraud: {1 - fraud_prob:.2f}\n(Demo Mode)"
                    self.result_label.config(text=result_text, foreground="green")

    def train_model(self):
        """Train the model using the data from dados_com_fraudes.csv"""
        try:
            # Show a loading message
            self.result_label.config(text="Treinando o modelo", foreground="blue")
            self.root.update()

            # Check if the data file exists
            file_path = 'dados_com_fraudes.csv'
            if not os.path.exists(file_path):
                # Try with the other filename from the original code
                file_path = 'dados_com_fraudes (1).csv'
                if not os.path.exists(file_path):
                    messagebox.showerror("error de arquivo",
                                         "dados_com_fraudes.csv arquivo não encontrado no diretorio correto.")
                    self.result_label.config(text="falha no modelo de treino", foreground="red")
                    return

            # Load the data
            df = pd.read_csv(file_path)

            # Preprocessing
            # Remove unnamed column if it exists
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace=True)

            # Process categorical variables
            le = LabelEncoder()
            if 'sex' in df.columns and df['sex'].dtype == 'object':
                df['sex'] = le.fit_transform(df['sex'])

            if 'smoker' in df.columns and df['smoker'].dtype == 'object':
                df['smoker'] = le.fit_transform(df['smoker'])

            if 'region' in df.columns and df['region'].dtype == 'object':
                df['region'] = le.fit_transform(df['region'])

            if 'tipo_procedimento' in df.columns and df['tipo_procedimento'].dtype == 'object':
                df['tipo_procedimento'] = le.fit_transform(df['tipo_procedimento'])

            # Drop date column if it exists
            if 'data_sinistro' in df.columns:
                df.drop('data_sinistro', axis=1, inplace=True)

            # Prepare features and target
            if 'fraudulent' not in df.columns:
                messagebox.showerror("Data Error", "The 'fraudulent' column is missing in the dataset.")
                self.result_label.config(text="Model training failed. Invalid data.", foreground="red")
                return

            X = df.drop('fraudulent', axis=1)
            y = df['fraudulent']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Get feature importance
            feature_importances = pd.Series(self.model.feature_importances_, index=X.columns)
            feature_importances.sort_values(ascending=False, inplace=True)
            top_features = feature_importances.head(3).index.tolist()

            # Save the model
            with open('dental_fraud_model.pkl', 'wb') as file:
                pickle.dump(self.model, file)

            # Display results
            result_text = f"Model trained successfully!\nAccuracy: {accuracy:.2f}\nTop features: {', '.join(top_features)}"
            self.result_label.config(text=result_text, foreground="green")

            messagebox.showinfo("Training Complete",
                                f"Model trained with accuracy: {accuracy:.2f}\nModel saved as 'dental_fraud_model.pkl'")

        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred while training the model: {str(e)}")
            self.result_label.config(text=f"Model training failed: {str(e)}", foreground="red")


def analyze_dataset():
    """Function to perform exploratory data analysis on the dataset"""
    try:
        # Try different file names that might exist
        file_paths = ['dados_com_fraudes.csv', 'dados_com_fraudes (1).csv']

        df = None
        for path in file_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break

        if df is None:
            print("Dataset file not found. Analysis cannot be performed.")
            return

        # Basic dataset information
        print("Dataset Information:")
        print(df.info())

        print("\nDescriptive Statistics:")
        print(df.describe())

        print("\nChecking for missing values:")
        print(df.isnull().sum())

        # Remove unnamed column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

        # Standardize values for procedure types (as in the original code)
        procedimento_valores = {
            'Limpeza Dental': 100,
            'Aparelho Ortodôntico': 1300,
            'Extração dental': 400,
            'Canal': 800,
            'Aplicação de protese': 3000,
            'Panoramica': 200,
            'Cirurgia': 5000
        }

        # Feature preprocessing
        le = LabelEncoder()
        df['sex'] = le.fit_transform(df['sex'])
        df['smoker'] = le.fit_transform(df['smoker'])
        df['region'] = le.fit_transform(df['region'])
        df['tipo_procedimento'] = le.fit_transform(df['tipo_procedimento'])

        if 'data_sinistro' in df.columns:
            df.drop('data_sinistro', axis=1, inplace=True)

        # Train a model to demonstrate feature importance
        X = df.drop('fraudulent', axis=1)
        y = df['fraudulent']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances.sort_values(ascending=False, inplace=True)

        print("\nFeature Importance:")
        print(feature_importances)

        # Save model for future use
        with open('dental_fraud_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        print("\nModel saved as 'dental_fraud_model.pkl'")

    except Exception as e:
        print(f"Error during data analysis: {str(e)}")


def main():
    """Main function to run the application"""
    print("Starting Dental Insurance Fraud Detection System")
    print("1. Running dataset analysis...")
    analyze_dataset()

    print("\n2. Starting GUI application...")
    root = tk.Tk()
    app = DentalInsuranceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()