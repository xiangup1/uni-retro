from unicore.tasks import UnicoreTask
from unicore.data import UnicoreDataset, data_utils, iterators
from ..data import CustomizedUnicoreDataset
import logging
logger = logging.getLogger(__name__)


class CustomizedUnicoreTask(UnicoreTask):

    def get_dynamic_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug(
                "reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]
        else:
            logger.info("get EpochBatchIterator for epoch {}".format(epoch))

        assert isinstance(dataset, UnicoreDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = self.indices
        # FFFFFFF1

        # create mini-batches with given size constraints
        # batch_sampler = dataset.batch_by_size(
        #     indices,
        #     batch_size=batch_size,
        #     required_batch_size_multiple=required_batch_size_multiple,
        # )

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )
        # 先忽略 filter_indices_by_size 这一块
        num_tokens_fn = self.num_tokens
        num_tokens_vec = self.num_tokens_vec(indices)
        # print('test indices: ', indices, num_tokens_vec)
        # create mini-batches with given size constraints
        batch_sampler = CustomizedUnicoreDataset.batch_by_size_dynamic(dataset,
                                                                       indices,
                                                                       num_tokens_fn,
                                                                       num_tokens_vec,
                                                                       max_tokens=max_tokens,
                                                                       max_sentences=max_sentences,
                                                                       required_batch_size_multiple=required_batch_size_multiple,
                                                                       )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter