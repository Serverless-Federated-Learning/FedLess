from fedless.common.persistence.client_daos import ClientParameterDao


def clean_db(mongo_client, sessions):

    client_parameter_dao = ClientParameterDao(mongo_client)
    client_parameter_dao.delete_all_except(sessions)
