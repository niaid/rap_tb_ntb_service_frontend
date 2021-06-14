import requests
import traceback
import os
import SimpleITK as sitk
import pandas as pd
import numpy as np
import inspect
from sklearn import metrics
import matplotlib.pyplot as plt

from PySide2.QtWidgets import (
    QMainWindow,
    QErrorMessage,
    QSlider,
    QWidget,
    QApplication,
    QFileDialog,
    QTextEdit,
    QLabel,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QScrollArea,
    QComboBox,
    QAction,
    QSplitter,
    qApp,
    QProgressDialog,
    QTextBrowser,
)
from PySide2.QtCore import Qt, QObject, QRunnable, Signal, QThreadPool, QThread
from PySide2.QtGui import QPixmap, QImage, QIcon
import PySide2.QtGui
import qdarkstyle
from docutils.core import publish_string


class ExistingFileOrDirectoryDialog(QFileDialog):
    """
    Custom file dialog for selecting either a directory or a file. The standard
    Qt dialogs only allow selection of file or directory not both.
    """

    def __init__(self, parent):
        super(ExistingFileOrDirectoryDialog, self).__init__()
        self.setOption(QFileDialog.DontUseNativeDialog)
        self.setFileMode(QFileDialog.Directory)
        self.currentChanged.connect(self._selected)

    def _selected(self, name):
        if os.path.isdir(name):
            self.setFileMode(QFileDialog.Directory)
        else:
            self.setFileMode(QFileDialog.ExistingFile)


class HelpDialog(QWidget):
    """
    Dialog for displaying a single html page with text converted from
    a reStructuredText (rst) string. The string is usually a class's
    docstring documentation. As the string may contain code examples
    the docutils package uses the pygments library for syntax highlighting.
    Depending on the GUI theme the default pygments CSS may not be appropriate
    (e.g. white text on white background). To use a specific style you need
    to generate a CSS which will be given as input to the set_rst_text:
    pygmentize -S monokai -f html -a pre.code > monokai.css
    To see available styles:
    pygmentize -L styles
    """

    def __init__(self, w=500, h=600):
        super(HelpDialog, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        # Use the QTextBrowser and configure it to allow clickable hyperlinks.
        self.help_text_edit = QTextBrowser()
        self.help_text_edit.setOpenLinks(True)
        self.help_text_edit.setOpenExternalLinks(True)
        layout.addWidget(self.help_text_edit)
        self.resize(w, h)

    def set_rst_text(self, txt, pygments_css_file_name=None):
        # Use docutils publish_string method to convert the rst to html.
        # Then insert the css into the html and display that.
        html_str = publish_string(txt, writer_name="html").decode("utf-8")
        if pygments_css_file_name:
            style_idx = html_str.index("</style>")
            with open(pygments_css_file_name, "r") as fp:
                css_str = fp.read()
                html_str = html_str[:style_idx] + css_str + html_str[style_idx:]
        self.help_text_edit.setHtml(html_str)


class TBorNotTBDialog(QMainWindow):
    """
    TB or not-TB
    ============

    This program allows you to classify **frontal chest x-ray images**, anterior-posterior
    or posterior-anterior, into one of two classes, TB or not-TB, using the
    `NIAID <https://www.niaid.nih.gov/>`_ TB
    Portals web service. The program supports
    all of the image formats supported by the `SimpleITK <https://simpleitk.org/>`_ image
    analysis toolkit (DICOM, jpg, png, tiff...).

    Additional `information on the NIAID TB Portals program is available
    online <https://tbportals.niaid.nih.gov/>`_.

    Application Settings
    --------------------

    The application allows you to select the specific algorithm to call, and set the
    query timeout and number of retries. Currently there are only two available algorithms,
    one using a single deep learning convolutional neural network model, DenseNet121,
    and another that uses multiple models, an ensemble of DenseNet121 networks.

    Selecting Input
    ---------------

    Data is loaded in one of two ways:

    1. Selection of a directory - directory and all its subdirectories are scanned for
       images which are used as input.
    2. Selection of a comma-separated-values (csv) file - the file is expected to have a column
       containing the paths to image files. The column header has a **required name,
       'file'**. If the file also has a column header named 'actual', that column
       is expected to have entries of the form TB or NOT_TB and will be used to
       evaluate the results obtained from the service.

       The contents of a sample input csv file are displayed below::

         file,actual
         data/px22.jpg,TB
         data/nx42.jpg,NOT_TB
         data/p8.dcm,TB
         data/n17.dcm,NOT_TB
         
       Once the data is loaded, clicking on the thumbnail image will display it in
       full resolution alongside the currently avaialble information.

    Query Service
    -------------
    After loading the data you can select a subset of the images that will be sent
    to the service for classification, using the checkbox associated with each of them. Once
    the service query is initiated the border of the thumbnail image indicates
    the status:

    1. Gray - never used in a query.
    2. Blue - query initiated for that image, no results received.
    3. Green - query performed, image succesfully processed and results available.
    4. Red -  query performed, failure.

    Save Results
    ------------

    You can save the currently available results at any time using the `Save results`
    menu item. This saves the raw results in a csv file. The raw results are the values
    returned from the TB/not-TB service. These include the `probability of TB`, the service's
    `decision` and additional information.

    If the input was in the form of a csv file which also contained the known class then
    you can `Save results and evaluation`. This will save the raw results and evaluate the
    service performance. The output includes:

    #. all_results.csv - A csv with the raw results from the service, includes all images.
    #. valid_results_used_in_evaluation.csv - A csv containing only the valid
       results which were used in the evaluation.
    #. evaluation_results.csv - A csv containing the evaulation results.
       Evaluation is performed using the services decision and a decision obtained
       when using the query dataset's optimal threshold, Youden index = argmax(sensitivity+specificity - 1). The latter appears
       in parenthesis:

       a. Accuracy: (TP+TN)/(P+N)
       b. Precision: TP/(TP+FP)
       c. Recall/Sensitivity: TP/(TP+FN)
       d. Specificity: TN/(TN+FP)
       e. F1: 2TP/(2TP+FP+FN)
       f. Area under the receiver operating curve.

    #. roc.pdf - ROC curve plot for all algorithms.
    #. confusion_matrix*.pdf - Confusion matrices for all algorithms using the
       service's decision and the decision based on the query dataset's optimal
       threshold.

    """

    def __init__(self):
        super(TBorNotTBDialog, self).__init__()

        # Disable the SimpleITK warnings (mostly the "Converting from MONOCHROME1 to MONOCHROME2"
        # which is common for our DICOM images)
        sitk.ProcessObject.GlobalWarningDisplayOff()

        # Use QT's global threadpool, documentation says: "This global thread pool
        # automatically maintains an optimal number of threads based on the
        # number of cores in the CPU."
        self.threadpool = QThreadPool.globalInstance()

        # csv file column titles
        self.csv_filename_column_title = "file"
        self.csv_actual_value_column_title = "actual"
        self.csv_service_response_column_title = "service_response"
        self.csv_file_size_column_title = "file_size_bytes"

        # service labels
        self.positive_label = "TB"
        self.negative_label = "NOT_TB"

        self.number_of_retries_range = [1, 5]
        self.default_number_of_retries = 3
        self.timeout_range = [1, 60]
        self.default_timeout = 5
        self.endpoint = "https://ria.niaidawsdev.net/TBorNotTB"
        self.algorithms = {"DenseNet121": "best", "DenseNet121 Ensemble": "ensemble"}

        # Size of thumbnails to use in the GUI
        self.thumbnail_size = [128] * 2

        # Limit the stylesheet to the pushbutton. Otherwise, the tooltip will inherit it and
        # we don't want that because the tooltip is long and won't display correctly.
        self.initial_button_stylesheet = "QPushButton{{qproperty-iconSize: {0}px {0}px; height : {0}px; width : {0}px; max-width: {0}px; border: 2px solid gray;}}".format(  # noqa: E501
            self.thumbnail_size[0]
        )
        self.waiting_button_stylesheet = "QPushButton{{qproperty-iconSize: {0}px {0}px; height : {0}px; width : {0}px; max-width: {0}px; border: 2px solid blue;}}".format(  # noqa: E501
            self.thumbnail_size[0]
        )
        self.finished_success_button_stylesheet = "QPushButton{{qproperty-iconSize: {0}px {0}px; height : {0}px; width : {0}px; max-width: {0}px; border: 2px solid green;}}".format(  # noqa: E501
            self.thumbnail_size[0]
        )
        self.finished_failure_button_stylesheet = "QPushButton{{qproperty-iconSize: {0}px {0}px; height : {0}px; width : {0}px; max-width: {0}px; border: 2px solid red;}}".format(  # noqa: E501
            self.thumbnail_size[0]
        )

        self.data_loader = DataLoader(self.thumbnail_size)
        self.data_loader.progress_signal.connect(self.__file_processed)
        self.data_loader.finished.connect(self.__load_data_finished)

        # Configure the help dialog.
        self.help_dialog = HelpDialog(w=700, h=500)
        self.help_dialog.setWindowTitle("TB or Not TB")
        self.help_dialog.set_rst_text(inspect.getdoc(self))

        self.__create_gui()
        self.setWindowTitle("TB or Not TB - That is the Question")
        self.show()

    def __rap_tb_ntb_query(self, url_name, algorithm_name, timeout):
        """
        Wrapper function which is specific to the NIAID service. It returns
        a generic function which only expects a file name. This potentially enables
        expanding the front end to other services with different RESTful APIs
        in an easy manner.
        """

        def tb_ntb_query(file_name):
            error_message = ""
            results_dict = {}
            try:
                response = requests.post(
                    url_name,
                    files={"image": open(file_name, "rb")},
                    data={"method": algorithm_name},
                    timeout=timeout,
                )
                if response.status_code == requests.codes.ok:
                    json_data = response.json()
                    for key in json_data:
                        results_dict[f"{algorithm_name}: {key}"] = json_data[key]
                else:
                    error_message = response.reason + f" ({response.status_code})."
            except requests.Timeout:
                error_message = "Request timed out."
            except requests.ConnectionError:
                error_message = "Connection error."
            except requests.exceptions.RequestException:  # anything else
                error_message = "Problem with service."
            return (error_message, results_dict)

        return tb_ntb_query

    def __error_function(self, message):
        error_dialog = QErrorMessage(self)
        # The QErrorMessage dialog automatically identifies if text is rich text,
        # html or plain text. Unfortunately, it doesn't do a good job when some of
        # the text is describing Exceptions due to number comparisons that include
        # the '>' symbol. As all invocations of this function are done with plain
        # text we use the convertToPlainText method to ensure that it is displayed
        # correctly.
        error_dialog.showMessage(PySide2.QtGui.Qt.convertFromPlainText(message))

    def __sitkimage_2_qpixmap(self, sitk_image):
        """
        Convert a SimpleITK.Image to Qt's QPixmap. Works with grayscale or a three
        channel image where we assume that it represents values in the RGB color space.
        In SimpleITK there is no notion of color space, so if the three channels were in
        HSV colorspace the display will look strange.
        """
        if (
            sitk_image.GetNumberOfComponentsPerPixel() == 1
            and sitk_image.GetPixelID() != sitk.sitkUInt8
        ):
            sitk_image = sitk.Cast(sitk.RescaleIntensity(sitk_image), sitk.sitkUInt8)
        arr = sitk.GetArrayViewFromImage(sitk_image)
        return QPixmap.fromImage(
            QImage(
                arr.data.tobytes(),
                sitk_image.GetWidth(),
                sitk_image.GetHeight(),
                arr.strides[0],  # number of bytes per row
                QImage.Format_Grayscale8
                if sitk_image.GetNumberOfComponentsPerPixel() == 1
                else QImage.Format_RGB888,
            )
        )

    def __create_gui(self):
        """
        Create the actual GUI and its layout (menu items, etc).
        """
        self.settings_dialog = self.__create_settings_widget()

        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        file_menu = menu_bar.addMenu("&File")
        settings_action = QAction("&Settings", self)
        settings_action.triggered.connect(self.settings_dialog.show)
        file_menu.addAction(settings_action)

        load_action = QAction("&Load data", self)
        load_action.triggered.connect(self.__load_data_and_config)
        file_menu.addAction(load_action)

        self.query_action = QAction("&Query service", self)
        self.query_action.triggered.connect(self.__call_service)
        file_menu.addAction(self.query_action)
        self.query_action.setEnabled(False)

        self.save_action = QAction("&Save results", self)
        self.save_action.triggered.connect(self.__save_results)
        file_menu.addAction(self.save_action)
        self.save_action.setEnabled(False)

        self.save_evaluate_action = QAction("&Save results and evaluation", self)
        self.save_evaluate_action.triggered.connect(self.__save_results_and_evaluation)
        file_menu.addAction(self.save_evaluate_action)
        self.save_evaluate_action.setEnabled(False)

        quit_action = QAction("&Quit", self)
        quit_action.triggered.connect(qApp.quit)
        file_menu.addAction(quit_action)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.help_dialog.show)
        menu_bar.setCornerWidget(self.help_button, Qt.TopRightCorner)

        central_widget = QWidget(self)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        main_widget = self.__create_main_widget()
        layout.addWidget(main_widget)

    def __file_processed(self):
        """
        Callback invoked when the service returns a response. Faliure and success
        are treated in the same manner.
        """
        self.num_files_loaded = self.num_files_loaded + 1
        self.progress_dialog.setValue(
            int(0.5 + 100 * self.num_files_loaded / self.all_df.shape[0])
        )

    def __create_main_widget(self):
        """
        Creation and layout of main GUI.
        """
        wid = QSplitter()

        left_widget = QWidget()
        wid.addWidget(left_widget)
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        input_wid = QWidget()
        self.input_layout = QVBoxLayout()
        input_wid.setLayout(self.input_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidget(input_wid)
        scroll_area.setWidgetResizable(True)
        left_layout.addWidget(scroll_area)

        self.all_checkboxes = []
        self.all_buttons = []

        checkbox = QCheckBox("select all")
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.__select_all_changed)
        left_layout.addWidget(checkbox)

        right_widget = QSplitter()
        right_widget.setOrientation(Qt.Vertical)
        wid.addWidget(right_widget)

        self.selected_image_label = QLabel()
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.selected_image_label)
        scroll_area.setWidgetResizable(True)
        right_widget.addWidget(scroll_area)

        info_widget = QWidget()
        right_widget.addWidget(info_widget)
        info_layout = QVBoxLayout()
        info_widget.setLayout(info_layout)
        info_layout.addWidget(QLabel("Image Information:"))
        self.selected_image_information_edit = QTextEdit()
        self.selected_image_information_edit.setReadOnly(True)
        info_layout.addWidget(self.selected_image_information_edit)

        return wid

    def __load_data_and_config(self):
        dialog = ExistingFileOrDirectoryDialog(self)
        if not dialog.exec():
            return

        # Get the input file names, either from a csv file or from the subdirectories
        # under the data root directory.
        self.num_files_loaded = 0
        self.progress_dialog = QProgressDialog("Loading data...", None, 0, 100)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        # Display the progress dialog only if its estimated operation duration is
        # more than 2000ms.
        self.progress_dialog.setMinimumDuration(2000)

        self.all_df = self.__get_file_names_df(dialog.selectedFiles()[0])
        if self.all_df.empty or (
            self.csv_filename_column_title not in self.all_df.columns
        ):
            self.__error_function(
                "No data loaded (error in csv file or empty directory)."
            )
        else:
            self.data_loader.reset()
            self.data_loader.file_names_list = list(
                self.all_df[self.csv_filename_column_title]
            )
            self.data_loader.start()

    def __load_data_finished(self):
        problematic_row_indexes = self.all_df.index[self.data_loader.error_indexes]
        problematic_files = list(
            self.all_df[self.csv_filename_column_title][problematic_row_indexes]
        )
        self.all_df.drop(problematic_row_indexes, inplace=True)
        self.all_df[self.csv_service_response_column_title] = "not computed"
        self.all_df[self.csv_file_size_column_title] = float("nan")
        if problematic_files:
            self.__error_function(
                "The following files were not loaded:\n {0}".format(
                    "\n".join(problematic_files)
                )
            )
        if not self.all_df.empty:
            self.selected_image_row_index = -1
            self.selected_image_label.clear()
            self.selected_image_information_edit.clear()
            # Remove all widgets from previous dataset, if any
            for cb, b in zip(self.all_checkboxes, self.all_buttons):
                cb.setParent(None)
                b.setParent(None)
            for i, image, file_name in zip(
                range(len(self.data_loader.images)),
                self.data_loader.images,
                self.all_df[self.csv_filename_column_title],
            ):
                layout = QHBoxLayout()
                self.input_layout.addLayout(layout)
                self.all_checkboxes.append(QCheckBox())
                self.all_checkboxes[-1].setChecked(True)
                layout.addWidget(self.all_checkboxes[-1])
                pixmap = QPixmap(
                    QImage(
                        sitk.GetArrayViewFromImage(image),
                        image.GetWidth(),
                        image.GetHeight(),
                        QImage.Format_Grayscale8
                        if image.GetNumberOfComponentsPerPixel() == 1
                        else QImage.Format_RGB888,
                    )
                )
                icon = QIcon()
                icon.addPixmap(pixmap, QIcon.Normal, QIcon.Off)
                self.all_buttons.append(QPushButton())
                self.all_buttons[-1].setStyleSheet(self.initial_button_stylesheet)
                self.all_buttons[-1].setIcon(icon)
                self.all_buttons[-1].setToolTip(file_name)
                self.all_buttons[-1].clicked.connect(self.__image_clicked(i))
                layout.addWidget(self.all_buttons[-1])
                layout.addStretch()
            self.save_action.setEnabled(True)
            # The reference labels are available so we can evaluate performance
            if self.csv_actual_value_column_title in self.all_df.columns:
                self.save_evaluate_action.setEnabled(True)
            self.query_action.setEnabled(True)
        # clear all data from the loader.
        self.data_loader.reset()

    def __image_clicked(self, image_row_index):
        def callback():
            image_info = self.all_df.iloc[image_row_index]
            image_file_reader = sitk.ImageFileReader()
            image_file_reader.SetFileName(image_info[self.csv_filename_column_title])
            image_file_reader.ReadImageInformation()
            image_size = list(image_file_reader.GetSize())
            if len(image_size) == 3:
                image_size[2] = 0
                image_file_reader.SetExtractSize(image_size)
            image = image_file_reader.Execute()
            self.selected_image_row_index = image_row_index
            self.selected_image_label.setPixmap(self.__sitkimage_2_qpixmap(image))
            self.selected_image_information_edit.setText(
                "\n".join([f"{index}: {value}" for index, value in image_info.items()])
            )

        return callback

    def __select_all_changed(self):
        new_value = self.sender().isChecked()
        for cb in self.all_checkboxes:
            cb.setChecked(new_value)

    def __call_service(self):
        self.total_quries = 0
        for cb in self.all_checkboxes:
            if cb.isChecked():
                self.total_quries = self.total_quries + 1
            cb.setEnabled(False)
        self.remaining_quries = self.total_quries
        for i, cb, file_name in zip(
            range(len(self.all_checkboxes)),
            self.all_checkboxes,
            self.all_df[self.csv_filename_column_title],
        ):
            if cb.isChecked():
                self.all_df.at[
                    self.all_df.index[i], self.csv_file_size_column_title
                ] = os.stat(file_name).st_size
                query_service = QueryService(
                    int(self.retry_value_label.text()),
                    i,
                    self.__rap_tb_ntb_query(
                        self.endpoint,
                        self.algorithms[self.algorithm_combo.currentText()],
                        int(self.timeout_value_label.text()),
                    ),
                    file_name,
                )
                query_service.signals.finished.connect(self.__query_service_finished)
                self.all_buttons[i].setStyleSheet(self.waiting_button_stylesheet)
                self.threadpool.start(query_service)
        self.update()

    def __query_service_finished(self, final_results):
        df_row, error_message, results = final_results
        df_index = self.all_df.index[df_row]
        if error_message:
            self.all_buttons[df_row].setStyleSheet(
                self.finished_failure_button_stylesheet
            )
            self.all_df.at[df_index, self.csv_service_response_column_title] = (
                "Error - " + error_message
            )
        else:
            self.all_buttons[df_row].setStyleSheet(
                self.finished_success_button_stylesheet
            )
            self.all_df.at[df_index, self.csv_service_response_column_title] = "success"
            # First query means we need to create all the columns
            if self.remaining_quries == self.total_quries:
                for key in results:
                    if key not in self.all_df.columns:
                        self.all_df[key] = ""
            for key, val in results.items():
                self.all_df.at[df_index, key] = val
        self.remaining_quries = self.remaining_quries - 1
        if self.selected_image_row_index == df_row:
            image_info = self.all_df.iloc[df_row]
            self.selected_image_information_edit.setText(
                "\n".join([f"{index}: {value}" for index, value in image_info.items()])
            )
        if self.remaining_quries == 0:
            for cb in self.all_checkboxes:
                cb.setEnabled(True)
            QMessageBox().information(self, "Message", "Calculation completed.")

    def __save_results(self):
        output_file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "", "csv(*.csv)"
        )
        if output_file_name:
            self.all_df.to_csv(output_file_name, index=False)

    def __save_results_and_evaluation(self):
        output_dir_name = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if output_dir_name:
            self.all_df.to_csv(
                os.path.join(output_dir_name, "all_results.csv"), index=False
            )
            # Only analyze results that are complete for all evaluated algorithms
            all_valid_results_df = self.all_df.dropna()
            if all_valid_results_df.empty:
                self.__error_function(
                    "No complete valid results obtained from service."
                )
                return
            elif (
                all_valid_results_df[self.csv_actual_value_column_title].nunique() == 1
            ):
                self.__error_function(
                    "Valid data is unary, only saved raw data, performance analysis not done."
                )
                return
            if all_valid_results_df.shape[0] != self.all_df.shape[0]:
                self.__error_function(
                    f"Evaluation performed on partial dataset with valid results ({all_valid_results_df.shape[0]}/{self.all_df.shape[0]})."  # noqa: E501
                )

            valid_algorithm_names = []
            for _, algorithm_name in self.algorithms.items():
                if f"{algorithm_name}: probability_of_TB" in all_valid_results_df:
                    valid_algorithm_names.append(algorithm_name)
            if valid_algorithm_names:
                self.__evaluate_and_save(
                    valid_algorithm_names, all_valid_results_df, output_dir_name
                )

    def __evaluate_and_save(
        self, valid_algorithm_names, valid_results_df, output_dir_name
    ):
        evaluation_metrics = [
            ("Accuracy", metrics.accuracy_score),
            (
                "Precision",
                lambda y_true, y_pred: metrics.precision_score(
                    y_true, y_pred, pos_label=self.positive_label
                ),
            ),
            (
                "Recall",
                lambda y_true, y_pred: metrics.recall_score(
                    y_true, y_pred, pos_label=self.positive_label
                ),
            ),
            (
                "Specificity",
                lambda y_true, y_pred: metrics.recall_score(
                    y_true, y_pred, pos_label=self.negative_label
                ),
            ),
            (
                "F1",
                lambda y_true, y_pred: metrics.f1_score(
                    y_true, y_pred, pos_label=self.positive_label
                ),
            ),
        ]
        identity_classifier = type(
            "IdentityClassifier",
            (),
            {"predict": lambda i: i, "_estimator_type": "classifier"},
        )
        evaluation_results = []
        # Setup of the ROC figure, all ROC curves from the algorithms will plot
        # to this figure
        roc_figure = plt.figure()
        plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        for algo_name in valid_algorithm_names:
            algo_results = []
            fpr, tpr, thresholds = metrics.roc_curve(
                valid_results_df[self.csv_actual_value_column_title],
                valid_results_df[f"{algo_name}: probability_of_TB"],
                self.positive_label,
            )
            roc_auc = metrics.auc(fpr, tpr)
            youden_threshold = thresholds[np.argmax(tpr - fpr)]
            valid_results_df[
                f"{algo_name}: dataset_youden_threshold_decision({youden_threshold})"
            ] = valid_results_df[f"{algo_name}: probability_of_TB"].apply(
                lambda x, bound_threshold=youden_threshold: self.positive_label
                if x > bound_threshold
                else self.negative_label
            )
            # Plot the ROC for this algorithm
            plt.figure(roc_figure.number)
            plt.plot(
                fpr,
                tpr,
                lw=1,
                label=algo_name + f", AUC = {roc_auc:.3f}",
                linestyle="solid",
            )
            # Plot confusion matrix for this algorithm using the fixed/service threshold and the
            # optimal dataset specific threshold (Youden index)
            plt.figure()
            metrics.plot_confusion_matrix(
                identity_classifier,
                valid_results_df[f"{algo_name}: decision"],
                valid_results_df[self.csv_actual_value_column_title],
                cmap=plt.cm.Blues,
                colorbar=False,
            )
            plt.savefig(
                os.path.join(
                    output_dir_name, f"confusion_matrix_{algo_name}_fixed_threshold.pdf"
                ),
                bbox_inches="tight",
            )
            plt.figure()
            metrics.plot_confusion_matrix(
                identity_classifier,
                valid_results_df[
                    f"{algo_name}: dataset_youden_threshold_decision({youden_threshold})"
                ],
                valid_results_df[self.csv_actual_value_column_title],
                cmap=plt.cm.Blues,
                colorbar=False,
            )
            plt.savefig(
                os.path.join(
                    output_dir_name,
                    f"confusion_matrix_{algo_name}_youden_threshold.pdf",
                ),
                bbox_inches="tight",
            )
            # Evaluate results using all the common metrics
            for _, f in evaluation_metrics:
                using_service_threshold = f(
                    valid_results_df[self.csv_actual_value_column_title],
                    valid_results_df[f"{algo_name}: decision"],
                )
                using_dataset_optimal_threshold = f(
                    valid_results_df[self.csv_actual_value_column_title],
                    valid_results_df[
                        f"{algo_name}: dataset_youden_threshold_decision({youden_threshold})"
                    ],
                )
                algo_results.append(
                    f"{using_service_threshold} ({using_dataset_optimal_threshold})"
                )
            algo_results.append(roc_auc)
            evaluation_results.append(algo_results)
        # Finish plotting the ROC figure, add the legend and save
        plt.figure(roc_figure.number)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir_name, "roc.pdf"), bbox_inches="tight")
        # Save the data actually used in the computations
        valid_results_df.to_csv(
            os.path.join(output_dir_name, "valid_results_used_in_evaluation.csv"),
            index=False,
        )
        # Save the evaluation results
        column_names, _ = zip(*evaluation_metrics)
        column_names = column_names + ("ROC AUC",)
        evaluation_results_df = pd.DataFrame(
            data=evaluation_results, index=valid_algorithm_names, columns=column_names
        )
        evaluation_results_df.index.name = "Algorithm"
        evaluation_results_df.to_csv(
            os.path.join(output_dir_name, "evaluation_results.csv")
        )

    def __get_file_names_df(self, csv_or_dir_name):
        df = pd.DataFrame()
        if os.path.isdir(csv_or_dir_name):
            file_names = []
            for dir_name, subdir_names, f_names in os.walk(csv_or_dir_name):
                file_names += [
                    os.path.join(os.path.abspath(dir_name), fname) for fname in f_names
                ]
            df = pd.DataFrame(data=file_names, columns=[self.csv_filename_column_title])
        else:
            try:
                df = pd.read_csv(csv_or_dir_name)
                # Check if dataframe has actual classes and that they are as expected
                if self.csv_actual_value_column_title in df.columns:
                    # Remove leading/trailing whitespace, acommodate for minor user error in
                    # creation of input csv.
                    df[self.csv_actual_value_column_title] = df[self.csv_actual_value_column_title].str.strip()
                    if not df[self.csv_actual_value_column_title].isin([self.positive_label, self.negative_label]).all():
                        df = pd.DataFrame()
            except Exception:
                pass
        return df

    def __create_settings_widget(self):
        wid = QWidget()
        wid.setWindowTitle("Settings")
        input_layout = QVBoxLayout()
        wid.setLayout(input_layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Query timeout [sec]:"))
        timeout_slider = QSlider(orientation=Qt.Horizontal)
        timeout_slider.setMinimum(self.timeout_range[0])
        timeout_slider.setMaximum(self.timeout_range[1])
        timeout_slider.setValue(self.default_timeout)
        layout.addWidget(timeout_slider)
        self.timeout_value_label = QLabel()
        self.timeout_value_label.setNum(self.default_timeout)
        timeout_slider.valueChanged.connect(self.timeout_value_label.setNum)
        layout.addWidget(self.timeout_value_label)
        input_layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Number of retries:"))
        retry_slider = QSlider(orientation=Qt.Horizontal)
        retry_slider.setMinimum(self.number_of_retries_range[0])
        retry_slider.setMaximum(self.number_of_retries_range[1])
        retry_slider.setValue(self.default_number_of_retries)
        layout.addWidget(retry_slider)
        self.retry_value_label = QLabel()
        self.retry_value_label.setNum(self.default_number_of_retries)
        retry_slider.valueChanged.connect(self.retry_value_label.setNum)
        layout.addWidget(self.retry_value_label)
        input_layout.addLayout(layout)

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Algorithm:"))
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(list(self.algorithms.keys()))
        layout.addWidget(self.algorithm_combo)
        input_layout.addLayout(layout)

        button = QPushButton("Done")
        button.clicked.connect(wid.hide)
        input_layout.addWidget(button)
        return wid


class DataLoader(QThread):
    progress_signal = Signal()

    def __init__(self, thumbnail_size):
        super(DataLoader, self).__init__()
        self.thumbnail_size = thumbnail_size
        self.image_file_reader = sitk.ImageFileReader()
        self.reset()

    def reset(self):
        self.images = []
        self.error_indexes = []
        self.file_names_list = []

    def run(self):
        for i, file_name in enumerate(self.file_names_list):
            try:
                self.image_file_reader.SetFileName(file_name)
                self.image_file_reader.ReadImageInformation()
                num_components = self.image_file_reader.GetNumberOfComponents()
                # not a grayscale or color image
                if num_components not in [1, 3]:
                    self.error_indexes.append(i)
                else:
                    image_size = list(self.image_file_reader.GetSize())
                    # not a 2D image
                    if (
                        not (len(image_size) == 3 and image_size[2] == 1)
                        and not len(image_size) == 2
                    ):
                        self.error_indexes.append(i)
                    else:  # 2D image posing as a 3D one
                        if len(image_size) == 3 and image_size[2] == 1:
                            image_size[2] = 0
                        self.image_file_reader.SetExtractSize(image_size)
                        self.images.append(
                            self.__resize_and_scale_uint8(
                                self.image_file_reader.Execute(), self.thumbnail_size
                            )
                        )
                self.progress_signal.emit()
                # progress_dialog.setValue(int(0.5 + 100*i/number_of_files))
            except Exception:
                self.error_indexes.append(i)
                self.progress_signal.emit()

    def __resize_and_scale_uint8(self, image, new_size, outside_pixel_value=0):
        """
        Resize the given image to the given size, with isotropic pixel spacing
        and scale the intensities to [0,255].

        Resizing retains the original aspect ratio, with the original image centered
        in the new image. Padding is added outside the original image extent using the
        provided value.

        :param image: A SimpleITK image.
        :param new_size: List of ints specifying the new image size.
        :param outside_pixel_value: Value in [0,255] used for padding.
        :return: a 2D SimpleITK image with desired size and a pixel type of sitkUInt8
        """
        # Rescale intensities if scalar image with pixel type that isn't sitkUInt8.
        # We rescale first, so that the zero padding makes sense for all original image
        # ranges. If we resized first, a value of zero in a high dynamic range image may
        # be somewhere in the middle of the intensity range and the outer border has a
        # constant but arbitrary value.
        if (
            image.GetNumberOfComponentsPerPixel() == 1
            and image.GetPixelID() != sitk.sitkUInt8
        ):
            final_image = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
        else:
            final_image = image
        new_spacing = [
            ((osz - 1) * ospc) / (nsz - 1)
            for ospc, osz, nsz in zip(
                final_image.GetSpacing(), final_image.GetSize(), new_size
            )
        ]
        new_spacing = [max(new_spacing)] * final_image.GetDimension()
        center = final_image.TransformContinuousIndexToPhysicalPoint(
            [sz / 2.0 for sz in final_image.GetSize()]
        )
        new_origin = [
            c - c_index * nspc
            for c, c_index, nspc in zip(
                center, [sz / 2.0 for sz in new_size], new_spacing
            )
        ]
        final_image = sitk.Resample(
            final_image,
            size=new_size,
            outputOrigin=new_origin,
            outputSpacing=new_spacing,
            defaultPixelValue=outside_pixel_value,
        )
        return final_image


class QueryServiceSignals(QObject):
    finished = Signal(object)


class QueryService(QRunnable):
    def __init__(self, num_retries, df_row, query_function, file_name):
        super(QueryService, self).__init__()
        self.num_retries = num_retries
        self.df_row = df_row
        self.query_function = query_function
        self.file_name = file_name
        self.signals = QueryServiceSignals()
        self.results = {}
        self.error_message = ""

    def run(self):
        try:
            num_tries = 0
            no_response = True
            while num_tries < self.num_retries and no_response:
                self.error_message, self.results = self.query_function(self.file_name)
                num_tries = num_tries + 1
                if not self.error_message:
                    no_response = False
            self.signals.finished.emit((self.df_row, self.error_message, self.results))
        # Use the stack trace as the error message to provide enough
        # detailes for debugging.
        except Exception:
            self.error_message = (
                "Exception occurred during computation:\n" + traceback.format_exc()
            )
            self.signals.finished.emit((self.df_row, self.error_message, self.results))


if __name__ == "__main__":
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside2"))
    tb_ntb_dialog = TBorNotTBDialog()  # noqa: F841
    app.exec_()
