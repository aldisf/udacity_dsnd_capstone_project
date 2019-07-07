import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# For aesthetics :) 
starbucks_green = '#00653B'
starbucks_lightbrown = '#c79d69'
starbucks_darkbrown = '#6d5038'

def visualize_categorical_data(df, column, figsize=None, color=starbucks_green):
    '''
    This function provides a wrapper for plotting the frequency distribution
    of a particular column in a dataframe. Graphing will be done via sns.countplot
    and this function will provide the graph with necessary details (eg. axis labels and titles)
    
    Args
        df (pd.DataFrame) - dataframe containing data to be visualised
        column (str) - column name to be visualised from the dataframe
        figsize (tuple) - tuple of ints to modify the visualisation size
        color - color to be used in the visualisation
        
    Returns
        None
    '''
    plt.figure()
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.title('Frequency Distribution for {}'.format(column))
    sns.countplot(df[column], color=color);
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    


def compare_offers(offer_df, metrics_to_compare):
    '''
    Produces multiple distribution plots in 1 line showing the distribution of the metrics 
    inside metrics_to_compare
    
    Args
        offer_df (pd.DataFrame) - offer level dataframe to be visualised
        metrics_to_compare (list) - list of metrics to be compared across different instances
            of the same offer type
    
    Returns
        None
    
    '''
    fig, axs = plt.subplots(1, len(metrics_to_compare))
    fig.set_size_inches(28,4)

    for index, metric in enumerate(metrics_to_compare):
        ax = axs[index]
        ax.set_title(metric, fontsize=20)
        sns.countplot(offer_df[metric], ax=ax, color=starbucks_green)

    fig.tight_layout()

    plt.show()

    display(offer_df[['id'] + metrics_to_compare])
    
    

def get_tenure(date, reference_date, granularity='days'):
    '''
    Given a date and a reference_date, this function will return how 
    many time units specified in granularity has passed. 
    
    Reference date has to be later than date.
    
    Args
        date (int) - date in format of %Y%m%d
        reference_date (int) - reference date in format of %Y%m%d
        granularity (str) - unit of time to which the distance will be converted to.
            Accepted inputs : 'days' or 'months'.
    
    Returns
        tenure - the amount of unit of time that has passed between date and reference date.
            Months will be equivalent to np.ceil(days / 30)
    '''
    
    assert reference_date > date
    
    date_dt = datetime.strptime(str(date), '%Y%m%d')
    reference_date_dt = datetime.strptime(str(reference_date), '%Y%m%d')
    
    tenure = (reference_date_dt - date_dt).days
    
    if granularity == 'months':
        tenure = int(np.ceil(tenure / 30))
    
    return tenure



def ecdf(data, rounded=True):
    '''
    Returns the cumulative density distribution  of the data
    
    Args 
        data (List, Numpy array) - The data from which the cumulative distribution function is to be generated
        rounded (bool) - If True, the data will be rounded to the nearest integer before grouping to reduce
            the size of the output
    
    Returns
        x (Numpy array) - Ordered list of data
        y (Numpy array) - The corresponding cumulative distribution value 
    '''
    
    x = np.zeros(len(data))
    y = np.zeros(len(data))
    
    data_array = np.array(data)
    data_array.sort()
    
    for index, element in enumerate(data_array):
        x[index] = element
        y[index] = (index + 1) / len(data)
    
    if rounded:
        x = np.round(x, 0)
        
    temp_df = pd.DataFrame({
        'x' : x,
        'y' : y
    })
    
    temp_df = temp_df.groupby(['x'])['y'].max().reset_index()
    temp_df = temp_df.sort_values(by='x', ascending=True)
    
    x = temp_df['x'].values
    y = temp_df['y'].values
    
    return x, y
    
    
def clean_up_overlapping_views(df, result_dict):
    '''
    In order to deal with non-unique assignment of views to received offers,
    this function will help to assign 1-1 mapping of the problematic view events by following 
    the heuristic of: 
    
    If there are two of the same offer being received by a person, the newer one will be
    viewed first.
    
    Args
        df (pd.DataFrame) - DataFrame containing the following columns: 
            ['person', 'offer_id', 'receive_count', 'received_time', 'view_count', 'viewed_time']
    
    Returns
        None - The function will update the dictionary supplied in the result_dict argument instead. 
            This dictionary has to be defined prior to using this function via df.apply
        
    '''
    if df['receive_count'] == df['view_count']:
        for i in range(df['view_count']):
            result_dict['person'].append(df['person'])
            result_dict['offer_id'].append(df['offer_id'])
            result_dict['received_time'].append(df['received_time'][i])
            result_dict['viewed_time'].append(df['viewed_time'][i])
    else:
        for i in range(df['view_count']):
            result_dict['person'].append(df['person'])
            result_dict['offer_id'].append(df['offer_id'])
            result_dict['received_time'].append(df['received_time'][::-1][i])
            result_dict['viewed_time'].append(df['viewed_time'][i])


            
def clean_up_overlapping_completes(df, result_dict):
    '''
    In order to deal with non-unique assignment of completion to received offers,
    this function will help to assign 1-1 mapping of the problematic complete events by following 
    the heuristic of: 
    
    If there are two of the same offer being received by a person, the older one will be
    completed first.
    
    For each completed event, the function will also subset the received_time list only to those
    that have expiry_time greater than or equal the completed_time.
    
    assigned_offer variable below serves as a temporary list that will be used to check whether
    a particular received_time has been assigned to a certain completion event to facilitate
    1-1 mapping.
    
    Args
        df (pd.DataFrame) - DataFrame containing the following columns: 
            ['person', 'offer_id', 'receive_count', 'received_time', 'complete_count', 'completed_time']
    
    Returns
        None - The function will update the dictionary supplied in the result_dict argument instead. 
            This dictionary has to be defined prior to using this function via df.apply
        
    '''
    if df['receive_count'] == df['complete_count']:
        for i in range(df['complete_count']):            
            result_dict['person'].append(df['person'])
            result_dict['offer_id'].append(df['offer_id'])
            result_dict['received_time'].append(df['received_time'][i])
            result_dict['completed_time'].append(df['completed_time'][i])
    else:
        
        assigned_offer = []
        for i in range(df['complete_count']):
            complete_time = df['completed_time'][i]
            valid_offers_index = np.where(np.array(df['expiry_time']) >= complete_time)[0].min()
            valid_offers = df['received_time'][valid_offers_index:]
            
            for receive_time in valid_offers:
                if receive_time not in assigned_offer:
                    result_dict['received_time'].append(receive_time)
                    result_dict['completed_time'].append(complete_time)
                    assigned_offer.append(receive_time)
                    result_dict['person'].append(df['person'])
                    result_dict['offer_id'].append(df['offer_id'])
                    break
                else:
                    # Get the next earliest
                    continue



def get_offer_start_end_time(df, start_or_end):
    '''
    To determine a period in which a person is influenced by an offer, we will apply the following logic: 
    
    Start time: 
    1. If offer is viewed, return view_time
    2. If offer is not viewed, returns None
    
    End time
    1. If offer is viewed, minimum of expiry_time and completed_time
    2. If offer is not viewed, returns None
    
    Args
        df (pd.DataFrame) - offer-person level dataframe containing columns ['expiry_time','received_time','viewed_time','completed_time']
        start_or_end (str) - determine the type of period to be returned
    Returns
        time_bound (float) - the time bound associated with the type associated with start_or_end
    '''
    
    assert start_or_end in ('start','end','start_and_end')
    
    if pd.isnull(df['viewed_time']):
        return np.NaN
              
    else:
        start_time = df['viewed_time']
        
        if pd.isnull(df['completed_time']):
            end_time = df['expiry_time']
        else:
            end_time = np.min([df['expiry_time'], df['completed_time']])
            
        if start_or_end == 'start':
            return start_time

        elif start_or_end == 'end':
            return end_time

        else:
            return {'start' : start_time, 'end' : end_time}



def get_no_influence_periods(influence_array):
    '''
    Based on offers that are received by a person, this function will return the periods for which 
    the person is not influenced to complete an offer.
    
    This function is to be used with pd.Series.apply
    
    Args
        influence_array (list) - list of dictionaries containing the start and end of influence period
            as defined by get_offer_start_end_time function
    
    Returns
        periods (list) - list of dictionaries containing the start and end of no influence periods
    '''
    
    start_time = 0
    
    periods = []
    
    for influence in influence_array:
        if start_time < influence['start']:
            end_time = influence['start'] - 1
            periods.append({'start' : start_time, 'end' : end_time})
        
        start_time = influence['end'] + 1
    
    if len(periods) > 0:
        return periods
    else:
        return np.NaN


    
def get_no_influence_timebounds(experiment_df_with_no_influence, start_or_end):
    '''
    For a particular offer-person combination, this function will return the earliest no influence period of that particular person
    
    This function is to be used with df.apply 
    
    Args
        experiment_df_with_no_influence (pd.DataFrame) - pandas DataFrame containing the column containing no_influence_timebounds
            as the output of get_no_influence_periods function
        start_or_end (str) - Defines the type of return. start or end returns the timebound, start_and_end returns a dictionary with
            both start and end as key-value pair
            
    Returns
        no_influence_time_bound (str or dict) - The earliest time bound in which the person has no influence of offers
    '''
    
    assert start_or_end in ('start', 'end', 'start_and_end')
    
    influence_start_time = experiment_df_with_no_influence['viewed_time']
    no_influence_bounds = experiment_df_with_no_influence['no_influence_timebounds']
    
    if pd.isnull(influence_start_time) or type(no_influence_bounds) == float:
        return np.NaN
    
    else:
        no_influence_time_bound = False
        
        for no_influence in no_influence_bounds:
            if influence_start_time > no_influence['end']:
                no_influence_time_bound = no_influence
                break
            else:
                continue
    
    if not no_influence_time_bound:
        return np.NaN
    else:
        if start_or_end == 'start':
            return no_influence_time_bound['start']
        elif start_or_end == 'end':
            return no_influence_time_bound['end']
        else:
            return no_influence_time_bound


def prep_offer_data(users, transcript, portfolio):
    '''
    This function wraps the data cleaning procedure covered in part 2
    of the notebook. Returns the person-offer level dataframe with the 
    transactional summary added inside

    Args
        users (pd.DataFrame) - DataFrame of users
        transcript (pd.DataFrame) - DataFrame of events related to the offers and transactions
        portfolio (pd.DataFrame) - DataFrame listing all the portfolios that have been cleaned

    Returns 
        users_with_complete_data (pd.DataFrame) - List of users with complete data and additional features
            engineered inside the table
        experiment_df (pd.DataFrame) - Person-offer level dataframe containing the time of interactions
            and the transactions that happens within the particular period

    '''
    users['age'] = users['age'].apply(lambda x: np.NaN if x == 118 else x)

    users['no_nans'] = users.isnull().sum(axis=1)

    # Split the original dataframe into two, and proceed using entries with complete data
    users_with_complete_data = users[users['no_nans'] == 0]
    users_with_missing_data = users[users['no_nans'] == 3]


    # Drop helper column
    users_with_complete_data = users_with_complete_data.drop(['no_nans'], axis=1)
    users_with_missing_data = users_with_missing_data.drop(['no_nans'], axis=1)

    gender_df = pd.get_dummies(users_with_complete_data['gender'])

    users_with_complete_data = users_with_complete_data.join(gender_df).drop(['gender'], axis=1)


    # Change the column name to person to join further
    users_with_complete_data = users_with_complete_data.rename(columns={
        'id' : 'person'
    })
                                                                
    users_with_missing_data = users_with_missing_data.rename(columns={
        'id' : 'person'
    })


    # Get the user tenure
    reference_date = 20190101

    users_with_complete_data['tenure_days'] = users_with_complete_data['became_member_on']\
        .apply(get_tenure, reference_date=reference_date, granularity='days')

    users_with_complete_data['tenure_months'] = users_with_complete_data['became_member_on']\
        .apply(get_tenure, reference_date=reference_date, granularity='months')


    # Split the transcript dataframe to categories
    offer_received_df = transcript[transcript['event'] == 'offer received']
    offer_viewed_df = transcript[transcript['event'] == 'offer viewed']
    offer_completed_df = transcript[transcript['event'] == 'offer completed']
    txn_df = transcript[transcript['event'] == 'transaction']

    offer_received_df['offer_id'] = offer_received_df['value'].apply(lambda x: x['offer id'])
    offer_viewed_df['offer_id'] = offer_viewed_df['value'].apply(lambda x: x['offer id'])
    offer_completed_df['offer_id'] = offer_completed_df['value'].apply(lambda x: x['offer_id'])
    txn_df['amount'] = txn_df['value'].apply(lambda x: x['amount'])

    # Drop value and event column 
    offer_received_df = offer_received_df.drop(['value', 'event'], axis=1)
    offer_viewed_df = offer_viewed_df.drop(['value', 'event'], axis=1)
    offer_completed_df = offer_completed_df.drop(['value', 'event'], axis=1)
    txn_df = txn_df.drop(['value', 'event'], axis=1)


    # Rename time columns
    offer_received_df = offer_received_df.rename(columns={
        'time' : 'received_time'
    })

    offer_viewed_df = offer_viewed_df.rename(columns={
        'time' : 'viewed_time'
    })

    offer_completed_df = offer_completed_df.rename(columns={
        'time' : 'completed_time'
    })

    txn_df = txn_df.rename(columns={
        'time' : 'transaction_time'
    })


    txn_df = txn_df[txn_df['amount'] <= 50]

    offer_expiry = portfolio[['offer_id', 'duration', 'difficulty', 'reward','offer_type']]

    offer_received_df = offer_received_df.merge(offer_expiry, on=['offer_id'])

    # Get the expiry time of each experiment
    offer_received_df['expiry_time'] = offer_received_df['received_time']\
                                            + (offer_received_df['duration'] * 24)



    viewed_offers = offer_received_df.merge(offer_viewed_df, on=['person','offer_id'])

    viewed_offers = viewed_offers[(viewed_offers['viewed_time'] >= viewed_offers['received_time']) & 
                                (viewed_offers['viewed_time'] <= viewed_offers['expiry_time'])]


    viewed_offers = viewed_offers[['person','offer_id','received_time','viewed_time']]

    # Get the offer_view events that are assigned to more than one received event
    overlapping_views = viewed_offers.groupby(['person', 'offer_id', 'viewed_time'])['received_time'].count().sort_values(ascending=False).reset_index()

    overlapping_views = overlapping_views[overlapping_views['received_time'] > 1]

    # Dummy var to facilitate left joining
    overlapping_views['is_overlap'] = 1

    non_overlapping_views = viewed_offers.merge(overlapping_views[['person','offer_id','is_overlap']], on=['person','offer_id'], how='left')

    non_overlapping_views = non_overlapping_views[pd.isnull(non_overlapping_views['is_overlap'])].drop(['is_overlap'], axis=1)


    overlapping_ids = overlapping_views[['person', 'offer_id']].drop_duplicates()

    overlapping_ids_receive_events = offer_received_df.merge(overlapping_ids, on=['person','offer_id'])

    overlapping_ids_view_events = offer_viewed_df.merge(overlapping_ids, on=['person','offer_id'])
    overlapping_ids_view_events['dummy_count'] = 1


    overlapping_ids_receive_count = overlapping_ids_receive_events.groupby(['person','offer_id']).agg({
        'expiry_time' : 'count',
        'received_time' : lambda x : sorted([i for i in x])
    }).rename(columns={
        'expiry_time' : 'receive_count'
    }).reset_index()


    overlapping_ids_view_count = overlapping_ids_view_events.groupby(['person', 'offer_id']).agg({
        'dummy_count' : 'count',
        'viewed_time' : lambda x : sorted([i for i in x])
    }).rename(columns={
        'dummy_count' : 'view_count'
    }).reset_index()

    overlapping_ids_receive_view_count = overlapping_ids_receive_count.merge(overlapping_ids_view_count, on=['person', 'offer_id'])

    # Initialize a dictionary - to be formed as pandas dataframe
    overlapping_views_dictionary = {
        'person' : [],
        'offer_id' : [],
        'received_time' : [],
        'viewed_time' : []
    }

    overlapping_ids_receive_view_count.apply(clean_up_overlapping_views, result_dict=overlapping_views_dictionary, axis=1)

    cleaned_overlapping_views = pd.DataFrame(overlapping_views_dictionary)

    viewed_offers = pd.concat([non_overlapping_views, cleaned_overlapping_views])

    viewed_offers = viewed_offers[['person','received_time','offer_id','viewed_time']]


    completed_offers = offer_received_df.merge(offer_completed_df, on=['person', 'offer_id'])

    completed_offers = completed_offers[(completed_offers['completed_time'] >= completed_offers['received_time']) & 
                                    (completed_offers['completed_time'] <= completed_offers['expiry_time'])]

    completed_offers = completed_offers[['person','offer_id','received_time','completed_time']]

    # Get the offer_complete events that are assigned to more than one received event
    overlapping_completes = completed_offers.groupby(['person', 'offer_id', 'completed_time'])['received_time'].count().sort_values(ascending=False).reset_index()

    overlapping_completes = overlapping_completes[overlapping_completes['received_time'] > 1]

    # Dummy var to facilitate left joining
    overlapping_completes['is_overlap'] = 1

    non_overlapping_completes = completed_offers.merge(overlapping_completes[['person','offer_id','is_overlap']], on=['person','offer_id'], how='left')

    non_overlapping_completes = non_overlapping_completes[pd.isnull(non_overlapping_completes['is_overlap'])].drop(['is_overlap'], axis=1)


    overlapping_ids = overlapping_completes[['person', 'offer_id']].drop_duplicates()

    overlapping_ids_receive_events = offer_received_df.merge(overlapping_ids, on=['person','offer_id'])

    overlapping_ids_complete_events = offer_completed_df.merge(overlapping_ids, on=['person','offer_id'])
    overlapping_ids_complete_events['dummy_count'] = 1


    overlapping_ids_receive_count = overlapping_ids_receive_events.groupby(['person','offer_id']).agg({
        'duration' : 'count',
        'expiry_time' : lambda x : sorted([i for i in x]),
        'received_time' : lambda x : sorted([i for i in x]),
    }).rename(columns={
        'duration' : 'receive_count'
    }).reset_index()


    overlapping_ids_complete_count = overlapping_ids_complete_events.groupby(['person', 'offer_id']).agg({
        'dummy_count' : 'count',
        'completed_time' : lambda x : sorted([i for i in x])
    }).rename(columns={
        'dummy_count' : 'complete_count'
    }).reset_index()

    overlapping_ids_receive_complete_count = overlapping_ids_receive_count.merge(overlapping_ids_complete_count, on=['person', 'offer_id'])

    # Initialize a dictionary - to be formed as pandas dataframe
    overlapping_completes_dictionary = {
        'person' : [],
        'offer_id' : [],
        'received_time' : [],
        'completed_time' : []
    }

    overlapping_ids_receive_complete_count.apply(clean_up_overlapping_completes, result_dict=overlapping_completes_dictionary, axis=1)

    cleaned_overlapping_completes = pd.DataFrame(overlapping_completes_dictionary)

    completed_offers = pd.concat([non_overlapping_completes, cleaned_overlapping_completes])

    experiment_df = offer_received_df\
        .merge(viewed_offers, on=['person','received_time','offer_id'], how='left')

    experiment_df = experiment_df\
        .merge(completed_offers, on=['person', 'received_time', 'offer_id'], how='left')

    experiment_df['offer_influence_start_time'] = experiment_df.apply(get_offer_start_end_time, start_or_end = 'start', axis=1)
    experiment_df['offer_influence_end_time'] = experiment_df.apply(get_offer_start_end_time, start_or_end = 'end', axis=1)
    experiment_df['offer_influence_timebounds'] = experiment_df.apply(get_offer_start_end_time, start_or_end = 'start_and_end', axis=1)


    experiment_df = experiment_df.sort_values(by=['viewed_time', 'received_time'], ascending=[True, True])

    offer_influence_per_person = experiment_df.groupby(['person']).agg({
        'offer_influence_timebounds' : lambda x: [i for i in x if pd.notnull(i)]
    }).reset_index()

    offer_influence_per_person['no_influence_timebounds'] = offer_influence_per_person['offer_influence_timebounds'].apply(get_no_influence_periods)
    offer_influence_per_person.head()

    experiment_df = experiment_df.merge(offer_influence_per_person[['person', 'no_influence_timebounds']], on=['person'])

    experiment_df['no_influence_start_time'] = experiment_df.apply(get_no_influence_timebounds, start_or_end='start', axis=1)
    experiment_df['no_influence_end_time'] = experiment_df.apply(get_no_influence_timebounds, start_or_end='end', axis=1)

    experiment_df = experiment_df[['person', 'received_time', 'offer_id', 'offer_type', 'duration', 'reward', 'difficulty', 'expiry_time', 
                               'viewed_time', 'completed_time', 'offer_influence_start_time', 'offer_influence_end_time', 
                               'no_influence_start_time', 'no_influence_end_time']]

    cols_dict = {
        'rec_comp' : ['received_time', 'completed_time'],
        'influence' : ['offer_influence_start_time', 'offer_influence_end_time'],
        'no_influence' : ['no_influence_start_time', 'no_influence_end_time']
        }

    experiment_with_txn_df = experiment_df.merge(txn_df, on=['person'])


    pre_received = experiment_with_txn_df[['person', 'received_time', 'offer_id', 'transaction_time', 'amount']]

    pre_received = pre_received[pre_received['transaction_time'] < pre_received['received_time']]

    pre_received = pre_received.groupby(['person', 'offer_id', 'received_time'])\
        .agg({
        'transaction_time' : 'count',
        'amount' : 'sum'
    }).reset_index().rename(columns={
        'transaction_time' : 'pre_experiment_txn_count',
        'amount' : 'pre_experiment_sum_txn_value'
    })


    # Received and completed
    rec_comp_df = experiment_with_txn_df[['person', 'received_time', 'offer_id', 'completed_time', 
                                        'transaction_time', 'amount']]

    rec_comp_df = rec_comp_df[(rec_comp_df['transaction_time'] >= rec_comp_df['received_time']) &
                            (rec_comp_df['transaction_time'] <= rec_comp_df['completed_time'])]

    rec_comp_df = rec_comp_df.groupby(['person', 'received_time', 'offer_id','completed_time'])\
        .agg({
        'transaction_time' : 'count',
        'amount' : 'sum'})\
        .reset_index().rename(columns={
        'transaction_time' : 'rec_comp_txn_count',
        'amount' : 'rec_comp_sum_txn_value'})

    rec_comp_df['delta_days'] = rec_comp_df['completed_time'] - rec_comp_df['received_time']
    rec_comp_df['delta_days'] = rec_comp_df['delta_days'].apply(lambda x: 1 if x == 0 else x/24)

    rec_comp_df['rec_comp_txn_value_per_day'] = rec_comp_df['rec_comp_sum_txn_value'] / rec_comp_df['delta_days']
    rec_comp_df['rec_comp_avg_txn_value'] = rec_comp_df['rec_comp_sum_txn_value'] / rec_comp_df['rec_comp_txn_count']

    rec_comp_df = rec_comp_df.drop(['delta_days'], axis=1)


    # Offer Influence
    influence_df = experiment_with_txn_df[['person', 'received_time', 'offer_id', 
                                        'offer_influence_start_time', 'offer_influence_end_time', 'transaction_time', 'amount']]

    influence_df = influence_df[(influence_df['transaction_time'] >= influence_df['offer_influence_start_time']) &
                            (influence_df['transaction_time'] <= influence_df['offer_influence_end_time'])]

    influence_df = influence_df.groupby(['person', 'received_time', 'offer_id', 
                                        'offer_influence_start_time', 'offer_influence_end_time'])\
        .agg({
        'transaction_time' : 'count',
        'amount' : 'sum'})\
        .reset_index().rename(columns={
        'transaction_time' : 'influence_txn_count',
        'amount' : 'influence_sum_txn_value'})

    influence_df['delta_days'] = influence_df['offer_influence_end_time'] - influence_df['offer_influence_start_time']
    influence_df['delta_days'] = influence_df['delta_days'].apply(lambda x: 1 if x == 0 else x/24)

    influence_df['influence_txn_value_per_day'] = influence_df['influence_sum_txn_value'] / influence_df['delta_days']
    influence_df['influence_avg_txn_value'] = influence_df['influence_sum_txn_value'] / influence_df['influence_txn_count']

    influence_df = influence_df.drop(['delta_days'], axis=1)
                
    # No influence 
    no_influence_df = experiment_with_txn_df[['person', 'received_time', 'offer_id', 
                                        'no_influence_start_time', 'no_influence_end_time', 'transaction_time', 'amount']]

    no_influence_df = no_influence_df[(no_influence_df['transaction_time'] >= no_influence_df['no_influence_start_time']) &
                            (no_influence_df['transaction_time'] <= no_influence_df['no_influence_end_time'])]

    no_influence_df = no_influence_df.groupby(['person', 'received_time', 'offer_id', 
                                        'no_influence_start_time', 'no_influence_end_time'])\
        .agg({
        'transaction_time' : 'count',
        'amount' : 'sum'})\
        .reset_index().rename(columns={
        'transaction_time' : 'no_influence_txn_count',
        'amount' : 'no_influence_sum_txn_value'})

    no_influence_df['delta_days'] = no_influence_df['no_influence_end_time'] - no_influence_df['no_influence_start_time']
    no_influence_df['delta_days'] = no_influence_df['delta_days'].apply(lambda x: 1 if x == 0 else x/24)

    no_influence_df['no_influence_txn_value_per_day'] = no_influence_df['no_influence_sum_txn_value'] / no_influence_df['delta_days']
    no_influence_df['no_influence_avg_txn_value'] = no_influence_df['no_influence_sum_txn_value'] / no_influence_df['no_influence_txn_count']

    no_influence_df = no_influence_df.drop(['delta_days'], axis=1)

    experiment_df = experiment_df.merge(pre_received, on=['person', 'received_time', 'offer_id'], how='left')

    experiment_df = experiment_df.merge(rec_comp_df, on=['person','received_time','offer_id', 'completed_time'], how='left')

    experiment_df = experiment_df.merge(influence_df, on=['person', 'received_time', 'offer_id', 
                                                        'offer_influence_start_time', 'offer_influence_end_time'], how='left')

    experiment_df = experiment_df.merge(no_influence_df, on=['person','received_time', 'offer_id',
                                                            'no_influence_start_time', 'no_influence_end_time'], how='left')


    return users_with_complete_data, experiment_df