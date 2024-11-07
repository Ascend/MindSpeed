import os

from sqlalchemy import Column, Integer, String, UniqueConstraint, text, desc
from sqlalchemy import create_engine, Float
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()
BaseHistory = declarative_base()


class DataBase:
    def __init__(self, working_dir: str):
        db_uri_history = f'sqlite:///{os.path.join(working_dir, "operator_history.db")}'
        db_connection_history = DBConnection(db_uri_history)
        self.operator_history_dao = OperatorHistoryDAO(db_connection_history)
        BaseHistory.metadata.create_all(db_connection_history.engine)
        db_uri_different = f'sqlite:///{os.path.join(working_dir, "operator_different.db")}'
        db_connection_different = DBConnection(db_uri_different)
        self.operator_different_dao = OperatorHistoryDAO(db_connection_different)
        BaseHistory.metadata.create_all(db_connection_different.engine)
        db_uri_profiling = f'sqlite:///{os.path.join(working_dir, "operator_profiling.db")}'
        db_connection_profiling = DBConnection(db_uri_profiling)
        self.operator_profiling_dao = OperatorHistoryDAO(db_connection_profiling)
        BaseHistory.metadata.create_all(db_connection_profiling.engine)

    def insert_not_found_list(self, operator_list):
        operator_different_list = []
        for operator in operator_list:
            operator_different_list.append(operator[0].convert_to_dict())
        self.operator_different_dao.insert_history(operator_different_list)


class OperatorHistory(BaseHistory):
    __tablename__ = 'operator_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    types = Column(String)
    accelerator_core = Column(String)
    input_shape = Column(String)
    output_shape = Column(String)
    duration = Column(Float)
    device = Column(String)
    jit = Column(Integer)
    cann = Column(String)
    driver = Column(String)
    dtype = Column(String)
    reverse1 = Column(String)
    __table_args__ = (
        UniqueConstraint('types', 'accelerator_core', 'input_shape', 'output_shape', 'device', 'jit',
                         'cann', 'driver', 'dtype', name='unique_operator'),)

    def __init__(self, types, accelerator_core, input_shape, output_shape, duration, device, jit, cann, driver, dtype):
        self.types = types
        self.accelerator_core = accelerator_core
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.duration = duration
        self.device = device
        self.jit = jit
        self.cann = cann
        self.driver = driver
        self.dtype = dtype
        self.reverse1 = "None"

    def __str__(self):
        rt = []
        rt.append(f"{'Operator Types':<30}{str(self.types):<40}")
        rt.append(f"{'accelerator_core':<30}{str(self.accelerator_core):<40}")
        rt.append(f"{'input_shape':<30}{str(self.input_shape):<40}")
        rt.append(f"{'output_shape':<30}{str(self.output_shape):<40}")
        rt.append(f"{'duration':<30}{str(self.duration):<40}")
        rt.append(f"{'device':<30}{str(self.device):<40}")
        rt.append(f"{'jit':<30}{str(self.jit):<40}")
        rt.append(f"{'cann':<30}{str(self.cann):<40}")
        rt.append(f"{'driver':<30}{str(self.driver):<40}")
        rt.append(f"{'dtype':<30}{str(self.dtype):<40}")
        return "\n".join(rt)

    def convert_to_dict(self):
        return {
            'types': self.types,
            'accelerator_core': self.accelerator_core,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'duration': self.duration,
            'device': self.device,
            'jit': self.jit,
            'cann': self.cann,
            'driver': self.driver,
            'dtype': self.dtype,
            'reverse1': self.reverse1
        }


class OperatorHistoryDAO(object):
    def __init__(self, db_connection):
        self.db_connection = db_connection

    def insert_history(self, data_list):
        def insert_data(session, dict_list):
            for data in dict_list:
                update_query = text('''
                                            UPDATE operator_history
                                            SET duration = (duration + :duration) / 2
                                            WHERE types = :types AND accelerator_core = :accelerator_core AND input_shape = :input_shape AND
                                                  output_shape = :output_shape AND device = :device AND jit = :jit AND cann = :cann AND
                                                  driver = :driver AND dtype = :dtype
                                            ''')
                result = session.execute(update_query, data)
                if result.rowcount == 0:
                    query = text('''
                                                INSERT INTO operator_history
                                                (types, accelerator_core, input_shape, output_shape, duration, device, jit, cann, driver, dtype, reverse1)
                                                SELECT :types, :accelerator_core, :input_shape, :output_shape, :duration, :device, :jit, :cann, :driver, :dtype, :reverse1
                                                WHERE NOT EXISTS(
                                                    SELECT 1 FROM operator_history WHERE
                                                    types = :types AND accelerator_core = :accelerator_core AND input_shape = :input_shape AND
                                                    output_shape = :output_shape AND device = :device AND jit = :jit AND cann = :cann AND
                                                    driver = :driver AND dtype = :dtype
                                                )
                                        ''')
                    session.execute(query, data)
                    session.commit()

        self.db_connection.execute(insert_data, data_list)

    def get_by_types_and_input_shape(self, types, input_shape):
        def get(session, key1, key2):
            results = session.query(OperatorHistory).filter_by(types=key1, input_shape=key2).all()
            objects = [OperatorHistory(types=result.types,
                                       accelerator_core=result.accelerator_core,
                                       input_shape=result.input_shape,
                                       output_shape=result.output_shape,
                                       duration=result.duration,
                                       device=result.device,
                                       jit=result.jit,
                                       cann=result.cann,
                                       driver=result.driver,
                                       dtype=result.dtype) for result in results]
            return objects

        return self.db_connection.execute(get, types, input_shape)

    def get_by_types_and_accelerator_core(self, accelerator_core, types):
        def get(session, key1, key2):
            results = session.query(OperatorHistory).filter_by(accelerator_core=key1, types=key2).all()
            objects = [OperatorHistory(types=result.types,
                                       accelerator_core=result.accelerator_core,
                                       input_shape=result.input_shape,
                                       output_shape=result.output_shape,
                                       duration=result.duration,
                                       device=result.device,
                                       jit=result.jit,
                                       cann=result.cann,
                                       driver=result.driver,
                                       dtype=result.dtype) for result in results]
            return objects

        return self.db_connection.execute(get, accelerator_core, types)


class Operator(object):

    def __init__(self, name, types, accelerator_core, input_shape, output_shape, duration):
        self.name = name
        self.types = types
        self.accelerator_core = accelerator_core
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.duration = duration

    def print_to_string(self):
        print("name: {}, input_shape: {}, output_shape: {}, duration: {}".format(
            self.name,
            self.input_shape,
            self.output_shape,
            self.duration
        ))


class DBConnection:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

    def execute(self, func, *args, **kwargs):
        session = self.Session()
        try:
            result = func(session, *args, **kwargs)
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
