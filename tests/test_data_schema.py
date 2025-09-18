import pytest
import pandas as pd
from dataprocessing import DataSchema

@pytest.fixture
def base_dataframe():
    """A standard, resuable pandas DataFrame instance."""
    
    age = [18]
    income = [20]
    city = pd.Series(pd.Categorical(['london'], categories = ['london', 'paris'], ordered=False))
    rating = pd.Series(pd.Categorical(['low'], categories=['low','medium','high'], ordered=True))

    df = {
        'age': age,
        'income': income,
        'city': city,
        'rating': rating
    }

    return pd.DataFrame(df)

@pytest.fixture
def base_schema():
    """A standard, reusable DataSchema instance."""
    return DataSchema(
        continuous={'age', 'income'},
        nominal={'city': {'london', 'paris'}},
        ordinal={'rating': ['low', 'medium', 'high']}
    )

class TestDataSchema:
    
    @pytest.fixture
    def equal_schema(self):
        """A schema identical to base_schema for equality checks."""
        return DataSchema(
            continuous={'age', 'income'},
            nominal={'city': {'london', 'paris'}},
            ordinal={'rating': ['low', 'medium', 'high']}
        )

    @pytest.fixture
    def num_append_base_schema(self):
        """base_schema appended with continuous column"""
        return DataSchema(
            continuous={'age', 'income', 'expenses'},
            nominal={'city': {'london', 'paris'}},
            ordinal={'rating': ['low', 'medium', 'high']}
        )

    @pytest.fixture
    def unord_append_base_schema(self):
        """base_schema appended with nominal categorical column"""
        return DataSchema(
            continuous = {'age', 'income'},
            nominal = {
                'city': {'london', 'paris'},
                'ethnicity': {'asian', 'caucasian', 'hispanic'},
            },
            ordinal={'rating': ['low', 'medium', 'high']}
        )

    @pytest.fixture
    def ord_append_base_schema(self):
        """base_schema appended with nominal categorical column"""
        return DataSchema(
            continuous = {'age', 'income'},
            nominal = {
                'city': {'london', 'paris'},
            },
            ordinal={
                'rating': ['low', 'medium', 'high'],
                'position': ['intern', 'staff', 'executive']
            }
        )

    @pytest.fixture
    def superset_schema(self):
        """A schema that is a superset of base_schema."""
        return DataSchema(
            continuous={'age', 'income'},
            nominal={'city': {'london', 'paris', 'tokyo'}}, # Extra category
            ordinal={'rating': ['low', 'medium', 'high', 'premium']} # Extra category
        )

    @pytest.fixture
    def different_num_cols_schema(self):
        """A schema with different columns."""
        return DataSchema(
            continuous={'age', 'height'}, # Different continuous col
            nominal={'city': {'london', 'paris'}},
            ordinal={'rating': ['low', 'medium', 'high']}
        )

    @pytest.fixture
    def empty_schema(self):
        """An empty schema"""
        return DataSchema(
            continuous = set(),
            nominal = {},
            ordinal = {}
        )

    def test_init(self, base_schema):
        # Happy
        assert base_schema.continuous == {'age', 'income'}
        assert base_schema.nominal == {'city': {'london', 'paris'}}
        assert base_schema.ordinal == {'rating': ['low', 'medium', 'high']}

    def test_eq(self, base_schema, equal_schema):
        # Happy
        assert base_schema == equal_schema

    def test_le(self, base_schema, superset_schema):
        # Happy
        assert base_schema <= base_schema
        assert base_schema <= superset_schema

    def test_ge(self, base_schema, superset_schema):
        # Happy
        assert base_schema >= base_schema
        assert superset_schema >= base_schema

    def test_lt(self, base_schema, superset_schema):
        # Happy
        assert base_schema < superset_schema

    def test_gt(self, base_schema, superset_schema):
        # Happy
        assert superset_schema > base_schema

    def test_build(self, base_schema, base_dataframe):
        # Happy
        assert base_schema == DataSchema.build(base_dataframe)

    def test_copy(self, base_schema):
        # Happy
        assert base_schema == base_schema.copy()

    def test_empty(self, empty_schema):
        # Happy
        assert empty_schema == DataSchema.empty()

    def test_append_num(self, base_schema, num_append_base_schema):
        appended_base_schema = base_schema._append_num('expenses')
        # Happy
        assert appended_base_schema == num_append_base_schema
        assert appended_base_schema != base_schema

    def test_append_ord(self, base_schema, ord_append_base_schema):
        # Happy
        assert base_schema._append_ord('position', ['intern', 'staff', 'executive']) == ord_append_base_schema

    def test_append_unord(self, base_schema, unord_append_base_schema):
        # Happy
        assert base_schema._append_unord('ethnicity', {'asian', 'caucasian', 'hispanic'}) == unord_append_base_schema

    def test_del_col(self, num_append_base_schema, base_schema, different_num_cols_schema):
        # Happy
        assert base_schema == num_append_base_schema._del_col('expenses')
        assert base_schema._del_col('income') == different_num_cols_schema._del_col('height')


class TestDataPipe:
    pass

class TestDataPipeline:
    pass