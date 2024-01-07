from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField
from wtforms.validators import DataRequired, ValidationError

class RainPridictionForm(FlaskForm):
    rainfall = FloatField('Rainfall', validators=[DataRequired()], 
    render_kw={'class':"form-control", 'placeholder':"Rainfall"})

    cloud3pm = FloatField('Cloud3pm', validators=[DataRequired()], 
    render_kw={'class':"form-control", 'placeholder':"Cloud3pm"})

    cloud9am = FloatField('Cloud9am', validators=[DataRequired()],
    render_kw={'class':"form-control", 'placeholder':"Cloud9am"})

    humidity3pm = FloatField('Humidity3pm', validators=[DataRequired()],
    render_kw={'class':"form-control", 'placeholder':"Humidity3pm"})

    rainToday = StringField('RainToday', validators=[DataRequired()],
    render_kw={'class':"form-control", 'placeholder':"RainToday"})

    submit = SubmitField('Predict rainfall tomorrow')

    def validate_rainToday(self, field):
        if not (field.data == 'Yes' or field.data == 'No'):
            raise ValidationError('Value must be either "Yes" or "No"')

