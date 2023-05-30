"""
Function that takes in a tuple of numbers (e.g. (1,2,3, ...,n)) and returns back the description
from the Google Cloud BigQuery Database to connect medical codes to their descriptions.

INPUT: tuple of type int.

OUTPUT: pandas dataframe of type str.
"""

import google.cloud.bigquery as gbq
import pandas as pd


def DescriptionsBQ(codes):
    def SQL_query(codes):
        codes = str(codes)
        select_input = "code, count, concept_name"
        from_input = "`hdcmlmed.upload_file.Word_Count`"
        join_input = "`hdcmlmed.omop_lds.CONCEPT` ON `hdcmlmed.upload_file.Word_Count`.code = `hdcmlmed.omop_lds.CONCEPT`.concept_id"
        where_input = codes
        order_input = "count DESC"

        QUERY = f"SELECT {select_input} \
                FROM {from_input} \
                JOIN {join_input} \
                WHERE code in {where_input} \
                ORDER BY {order_input}"

        return QUERY

    client = gbq.Client(project="hdcmlmed")

    QUERY = SQL_query(codes)
    query_job = client.query(QUERY)

    # Waits for query to finish
    rows = query_job.result()

    list = []
    for row in rows:
        list.append(row.concept_name)
        # print(row.concept_name)

    description = pd.DataFrame(list)

    return description


# #Example pull
# codes = (2108115, 320128, 35605482)
# descriptions = DescriptionsBQ(codes)
# print(descriptions)
